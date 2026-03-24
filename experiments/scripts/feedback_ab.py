"""A/B test: does apical feedback help or hurt?

Trains 4 configs on BabyLM 100k, probes each:
1. No feedback (baseline): S1←X, S2←X
2. S2→S1 only: S1←S2, S2←X
3. S3→S2 only: S1←X, S2←S3
4. Full feedback: S1←S2, S2←S3

Measures S1 BPC and S2 word consistency to see if feedback improves
downstream representations.
"""

import contextlib
import io
import sys
import time

sys.path.insert(0, "src")

import numpy as np

from step.config import (
    _default_motor_config,
    _default_region2_config,
    _default_region3_config,
    _default_s1_config,
    make_motor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import RewardModulator, SurpriseTracker, ThalamicGate
from step.cortex.topology import Topology
from step.data import inject_eom_tokens, prepare_tokens_charlevel
from step.decoders.dendritic import DendriticDecoder
from step.encoders.positional import PositionalCharEncoder
from step.probes.bpc import BPCProbe
from step.probes.word_selectivity import WordSelectivityProbe


def build_model(alphabet, s2_to_s1=True, s3_to_s2=True):
    encoder = PositionalCharEncoder(alphabet, max_positions=8)
    s1_cfg = _default_s1_config()
    s1 = make_sensory_region(s1_cfg, encoder.input_dim, encoder.encoding_width)

    r2_cfg = _default_region2_config()
    s2 = make_sensory_region(r2_cfg, s1.n_l23_total * 4, seed=123)

    r3_cfg = _default_region3_config()
    s3 = make_sensory_region(r3_cfg, s2.n_l23_total * 8, seed=456)

    m1_cfg = _default_motor_config()
    m1 = make_motor_region(m1_cfg, s1.n_l23_total, seed=789)
    bg = BasalGanglia(s1_cfg.n_columns + 1)

    cortex = Topology(encoder)
    cortex.add_region("S1", s1, entry=True)
    cortex.add_region("S2", s2)
    cortex.add_region("S3", s3)
    cortex.add_region("M1", m1, basal_ganglia=bg)

    cortex.connect(
        "S1",
        "S2",
        "feedforward",
        buffer_depth=4,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    cortex.connect(
        "S2",
        "S3",
        "feedforward",
        buffer_depth=8,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    cortex.connect("S1", "M1", "feedforward")
    cortex.connect(
        "M1",
        "S1",
        "apical",
        thalamic_gate=ThalamicGate(),
        reward_modulator=RewardModulator(),
    )

    if s2_to_s1:
        cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    if s3_to_s2:
        cortex.connect("S3", "S2", "apical", thalamic_gate=ThalamicGate())

    return cortex, encoder, s1, s2, s3


def probe(cortex, s1, s2, s3, tokens):
    s1_dec = cortex._regions["S1"].dendritic_decoder
    s2_dec = DendriticDecoder(source_dim=s2.n_l23_total, n_segments=16, n_synapses=48)
    s3_dec = DendriticDecoder(source_dim=s3.n_l23_total, n_segments=16, n_synapses=48)
    s1_bpc, s2_bpc, s3_bpc = BPCProbe(), BPCProbe(), BPCProbe()
    s2_words = WordSelectivityProbe(s2.n_columns)
    s3_words = WordSelectivityProbe(s3.n_columns)

    snaps = {"S1": [], "S2": [], "S3": []}
    for name, region in [("S1", s1), ("S2", s2), ("S3", s3)]:
        orig = region.process

        def make_cap(n, r, o):
            def cap(enc):
                result = o(enc)
                snaps[n].append((r.active_columns.copy(), r.active_l23.copy()))
                return result

            return cap

        region.process = make_cap(name, region, orig)

    with contextlib.redirect_stdout(io.StringIO()):
        cortex.run(tokens, log_interval=99999)

    from step.data import EOM_TOKEN, STORY_BOUNDARY

    content = [(t, s) for t, s in tokens if t != EOM_TOKEN and t != STORY_BOUNDARY]

    prev_s1 = np.zeros(s1.n_l23_total, dtype=np.bool_)
    prev_s2 = np.zeros(s2.n_l23_total, dtype=np.bool_)
    prev_s3 = np.zeros(s3.n_l23_total, dtype=np.bool_)

    n = min(len(content), len(snaps["S1"]), len(snaps["S2"]), len(snaps["S3"]))
    for i in range(n):
        tid, tstr = content[i]
        if i > 0:
            s1_bpc.step(tid, prev_s1, s1_dec)
            s2_bpc.step(tid, prev_s2, s2_dec)
            s3_bpc.step(tid, prev_s3, s3_dec)
        s2_dec.observe(tid, prev_s2)
        s3_dec.observe(tid, prev_s3)
        s2_words.step(tstr, snaps["S2"][i][0])
        s3_words.step(tstr, snaps["S3"][i][0])
        prev_s1 = snaps["S1"][i][1]
        prev_s2 = snaps["S2"][i][1]
        prev_s3 = snaps["S3"][i][1]

    s2_sum = s2_words.summary()
    s3_sum = s3_words.summary()
    return {
        "s1_bpc": s1_bpc.bpc,
        "s2_bpc": s2_bpc.bpc,
        "s3_bpc": s3_bpc.bpc,
        "s2_consistent": s2_sum["consistent_words"],
        "s2_mean_j": s2_sum["mean_consistency"],
        "s3_consistent": s3_sum["consistent_words"],
        "s3_mean_j": s3_sum["mean_consistency"],
    }


def main():
    print("Loading BabyLM...")
    train_tokens = prepare_tokens_charlevel(100000, dataset="babylm")
    train_tokens = inject_eom_tokens(train_tokens, segment_length=200, speak_window=10)
    probe_tokens = prepare_tokens_charlevel(30000, dataset="babylm")
    alphabet = sorted({ch for _, ch in train_tokens if _ >= 0})

    configs = [
        ("no feedback", False, False),
        ("S2→S1 only", True, False),
        ("S3→S2 only", False, True),
        ("full feedback", True, True),
    ]

    results = []
    for name, s2s1, s3s2 in configs:
        print(f"\n{'=' * 60}")
        print(f"  {name} (S2→S1={s2s1}, S3→S2={s3s2})")
        print(f"{'=' * 60}")

        t0 = time.monotonic()
        cortex, _encoder, s1, s2, s3 = build_model(
            alphabet,
            s2_to_s1=s2s1,
            s3_to_s2=s3s2,
        )

        print("  Training...")
        with contextlib.redirect_stdout(io.StringIO()):
            cortex.run(train_tokens, log_interval=99999)
        train_t = time.monotonic() - t0

        print(f"  Trained in {train_t:.0f}s. Probing...")
        t0 = time.monotonic()
        r = probe(cortex, s1, s2, s3, probe_tokens)
        r["name"] = name
        r["train_time"] = train_t
        results.append(r)

        print(
            f"  S1 BPC: {r['s1_bpc']:.3f}  S2 BPC: {r['s2_bpc']:.3f}"
            f"  S3 BPC: {r['s3_bpc']:.3f}"
        )
        print(f"  S2 consistent words: {r['s2_consistent']} (J={r['s2_mean_j']:.3f})")
        print(f"  S3 consistent words: {r['s3_consistent']} (J={r['s3_mean_j']:.3f})")
        print(f"  ({time.monotonic() - t0:.0f}s probe)")

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Config':<18} {'S1 BPC':>7} {'S2 BPC':>7} "
        f"{'S3 BPC':>7} {'S2 cons':>8} {'S3 cons':>8}"
    )
    print("-" * 58)
    for r in results:
        print(
            f"{r['name']:<18} {r['s1_bpc']:7.3f} {r['s2_bpc']:7.3f} "
            f"{r['s3_bpc']:7.3f} {r['s2_consistent']:8d} {r['s3_consistent']:8d}"
        )

    # Highlight impact
    base = next(r for r in results if r["name"] == "no feedback")
    full = next(r for r in results if r["name"] == "full feedback")
    s1_delta = base["s1_bpc"] - full["s1_bpc"]
    direction = "improved" if s1_delta > 0 else "worsened"
    print(f"\nFull feedback vs none: S1 BPC {direction} by {abs(s1_delta):.3f}")


if __name__ == "__main__":
    main()
