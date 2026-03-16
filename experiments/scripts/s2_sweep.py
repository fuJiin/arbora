"""Sweep S2 architecture on BabyLM to find best word-level representations.

Trains fresh models with different S2 configurations, then probes each
for character BPC and word-level selectivity. Compares S2 vs S1.

Variables:
  - S2 n_columns: 32, 64, 128
  - S2 k_columns: scaled proportionally (4, 8, 16)
  - buffer_depth: 4, 8
  - burst_gate: True, False
"""

import contextlib
import io
import sys
import time

sys.path.insert(0, "src")

import numpy as np

from step.config import (
    CortexConfig,
    _default_s1_config,
    _default_region2_config,
    _default_motor_config,
    make_sensory_region,
    make_motor_region,
)
from step.cortex.topology import Topology
from step.cortex.modulators import SurpriseTracker, ThalamicGate, RewardModulator
from step.cortex.basal_ganglia import BasalGanglia
from step.data import prepare_tokens_charlevel, inject_eom_tokens
from step.decoders.dendritic import DendriticDecoder
from step.encoders.positional import PositionalCharEncoder
from step.probes.bpc import BPCProbe
from step.probes.word_selectivity import WordSelectivityProbe


def build_model(alphabet, s2_cols, s2_k, buffer_depth, burst_gate):
    """Build hierarchy with custom S2 config."""
    encoder = PositionalCharEncoder(alphabet, max_positions=8)
    s1_cfg = _default_s1_config()
    s1 = make_sensory_region(s1_cfg, encoder.input_dim, encoder.encoding_width)

    r2_cfg = _default_region2_config()
    r2_cfg.n_columns = s2_cols
    r2_cfg.k_columns = s2_k
    s2_input_dim = s1.n_l23_total * buffer_depth
    s2 = make_sensory_region(r2_cfg, s2_input_dim, seed=123)

    m1_cfg = _default_motor_config()
    m1 = make_motor_region(m1_cfg, s1.n_l23_total, seed=456)
    bg = BasalGanglia(s1_cfg.n_columns + 1)

    cortex = Topology(encoder)
    cortex.add_region("S1", s1, entry=True)
    cortex.add_region("S2", s2)
    cortex.add_region("M1", m1, basal_ganglia=bg)

    cortex.connect(
        "S1", "S2", "feedforward",
        buffer_depth=buffer_depth, burst_gate=burst_gate,
    )
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S1", "M1", "feedforward")
    cortex.connect("S1", "S2", "surprise", surprise_tracker=SurpriseTracker())
    # M1→S1 apical only if M1 and S2 have same L2/3 dim (share apical target)
    if m1.n_l23_total == s2.n_l23_total:
        cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("M1", "S1", "reward", reward_modulator=RewardModulator())

    return cortex, encoder, s1, s2


def probe(cortex, encoder, s1, s2, tokens):
    """Run probe on trained model, return summary dict."""
    s2_decoder = DendriticDecoder(
        source_dim=s2.n_l23_total, n_segments=16, n_synapses=48,
    )
    s1_bpc = BPCProbe()
    s2_bpc = BPCProbe()
    s1_words = WordSelectivityProbe(s1.n_columns)
    s2_words = WordSelectivityProbe(s2.n_columns)

    # Capture states via monkey-patch
    s2_snaps, s1_snaps = [], []
    orig_s2 = s2.process
    orig_s1 = s1.process

    def cap_s2(enc):
        r = orig_s2(enc)
        s2_snaps.append((s2.active_columns.copy(), s2.active_l23.copy()))
        return r

    def cap_s1(enc):
        r = orig_s1(enc)
        s1_snaps.append((s1.active_columns.copy(), s1.active_l23.copy()))
        return r

    s2.process = cap_s2
    s1.process = cap_s1

    with contextlib.redirect_stdout(io.StringIO()):
        cortex.run(tokens, log_interval=99999)

    s2.process = orig_s2
    s1.process = orig_s1

    # Analyze
    from step.data import EOM_TOKEN, STORY_BOUNDARY
    content = [(t, s) for t, s in tokens if t != EOM_TOKEN and t != STORY_BOUNDARY]
    s1_dec = cortex._regions["S1"].dendritic_decoder

    prev_s1_l23 = np.zeros(s1.n_l23_total, dtype=np.bool_)
    prev_s2_l23 = np.zeros(s2.n_l23_total, dtype=np.bool_)

    n = min(len(content), len(s1_snaps), len(s2_snaps))
    for i in range(n):
        tid, tstr = content[i]
        if i > 0:
            s1_bpc.step(tid, prev_s1_l23, s1_dec)
            s2_bpc.step(tid, prev_s2_l23, s2_decoder)
        s2_decoder.observe(tid, prev_s2_l23)
        s1_words.step(tstr, s1_snaps[i][0])
        s2_words.step(tstr, s2_snaps[i][0])
        prev_s1_l23 = s1_snaps[i][1]
        prev_s2_l23 = s2_snaps[i][1]

    s1_sum = s1_words.summary()
    s2_sum = s2_words.summary()

    return {
        "s1_bpc": s1_bpc.bpc,
        "s2_bpc": s2_bpc.bpc,
        "s2_beats_s1": s2_bpc.bpc < s1_bpc.bpc,
        "s1_consistent": s1_sum["consistent_words"],
        "s2_consistent": s2_sum["consistent_words"],
        "s1_mean_consistency": s1_sum["mean_consistency"],
        "s2_mean_consistency": s2_sum["mean_consistency"],
        "s2_selective": s2_sum["selective_columns"],
        "s2_mean_selectivity": s2_sum["mean_selectivity"],
        "s2_top_words": s2_sum["top_consistent"][:5],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-chars", type=int, default=100000)
    parser.add_argument("--probe-chars", type=int, default=30000)
    args = parser.parse_args()

    # Load data
    print("Loading BabyLM...")
    train_tokens = prepare_tokens_charlevel(args.train_chars, dataset="babylm")
    train_tokens = inject_eom_tokens(train_tokens, segment_length=200, speak_window=10)
    probe_tokens = prepare_tokens_charlevel(args.probe_chars, dataset="babylm")
    alphabet = sorted({ch for _, ch in train_tokens if _ >= 0})

    # Sweep configs: (name, s2_cols, s2_k, buffer_depth, burst_gate)
    configs = [
        ("32c/k4/buf4/burst",   32,  4, 4, True),   # baseline
        ("64c/k8/buf4/burst",   64,  8, 4, True),
        ("128c/k16/buf4/burst", 128, 16, 4, True),
        ("64c/k4/buf4/burst",   64,  4, 4, True),   # sparse k
        ("64c/k8/buf8/burst",   64,  8, 8, True),   # deeper buffer
        ("64c/k8/buf4/noburst", 64,  8, 4, False),  # no burst gate
    ]

    results = []
    for name, s2_cols, s2_k, buf_depth, burst in configs:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  S2: {s2_cols} cols, k={s2_k}, buf={buf_depth}, burst={burst}")
        print(f"{'='*60}")

        # Build and train
        t0 = time.monotonic()
        cortex, encoder, s1, s2 = build_model(
            alphabet, s2_cols, s2_k, buf_depth, burst,
        )
        print(f"  Training on {len(train_tokens)} tokens...")
        with contextlib.redirect_stdout(io.StringIO()):
            cortex.run(train_tokens, log_interval=99999)
        train_time = time.monotonic() - t0
        print(f"  Trained in {train_time:.0f}s")

        # Probe
        print(f"  Probing on {len(probe_tokens)} tokens...")
        t0 = time.monotonic()
        r = probe(cortex, encoder, s1, s2, probe_tokens)
        probe_time = time.monotonic() - t0
        r["name"] = name
        r["train_time"] = train_time
        results.append(r)

        bpc_cmp = "S2 WINS" if r["s2_beats_s1"] else "S1 wins"
        print(f"  S1 BPC: {r['s1_bpc']:.3f}  S2 BPC: {r['s2_bpc']:.3f}  ({bpc_cmp})")
        print(f"  S1 consistent words: {r['s1_consistent']}  S2: {r['s2_consistent']}")
        print(f"  S2 selective cols: {r['s2_selective']}  mean entropy: {r['s2_mean_selectivity']:.3f}")
        if r["s2_top_words"]:
            top = ", ".join(f"'{w}'({j:.2f})" for w, j, _ in r["s2_top_words"])
            print(f"  Top S2 words: {top}")
        print(f"  ({probe_time:.0f}s probe)")

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'S2 BPC':>7} {'S2>S1':>6} {'S2 cons':>8} {'S2 sel':>7} {'Time':>6}")
    print("-" * 65)
    for r in results:
        win = "YES" if r["s2_beats_s1"] else "no"
        print(
            f"{r['name']:<25} {r['s2_bpc']:7.3f} {win:>6} "
            f"{r['s2_consistent']:8d} {r['s2_selective']:7d} "
            f"{r['train_time']:5.0f}s"
        )


if __name__ == "__main__":
    main()
