"""Probe S3 representations: character decoding + topic-level selectivity.

Loads a checkpoint with S3 and runs BabyLM corpus through the hierarchy,
measuring:
1. S3 dendritic decoder BPC (vs S1 and S2 baselines)
2. S3 word-level selectivity (inherited from word_selectivity probe)
3. S3 segment-level consistency — do columns fire consistently within
   segments (text between story boundaries)?
"""

import contextlib
import io
import sys

sys.path.insert(0, "src")

import numpy as np

from step.config import (
    _default_s1_config,
    _default_region2_config,
    _default_region3_config,
    _default_motor_config,
    make_sensory_region,
    make_motor_region,
)
from step.cortex.topology import Topology
from step.cortex.modulators import SurpriseTracker, ThalamicGate, RewardModulator
from step.cortex.basal_ganglia import BasalGanglia
from step.data import EOM_TOKEN, STORY_BOUNDARY, prepare_tokens_charlevel
from step.decoders.dendritic import DendriticDecoder
from step.encoders.positional import PositionalCharEncoder
from step.probes.bpc import BPCProbe
from step.probes.word_selectivity import WordSelectivityProbe


def build_model(alphabet):
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

    cortex.connect("S1", "S2", "feedforward", buffer_depth=4, burst_gate=True)
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S1", "S2", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S2", "S3", "feedforward", buffer_depth=8, burst_gate=True)
    cortex.connect("S2", "S3", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S3", "S2", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S1", "M1", "feedforward")
    if m1.n_l23_total == s2.n_l23_total:
        cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("M1", "S1", "reward", reward_modulator=RewardModulator())

    return cortex, encoder, s1, s2, s3


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Probe S3 representations")
    parser.add_argument("--checkpoint", default="babylm_s3_100k")
    parser.add_argument("--chars", type=int, default=30000)
    args = parser.parse_args()

    # Load data
    print("Loading BabyLM...")
    vocab_tokens = prepare_tokens_charlevel(100000, dataset="babylm")
    alphabet = sorted({ch for _, ch in vocab_tokens if _ >= 0})
    tokens = prepare_tokens_charlevel(args.chars, dataset="babylm")

    # Build model and load checkpoint
    cortex, encoder, s1, s2, s3 = build_model(alphabet)
    ckpt_path = f"experiments/checkpoints/{args.checkpoint}.ckpt"
    print(f"Loading checkpoint: {ckpt_path}")
    cortex.load_checkpoint(ckpt_path)

    # Create probes
    s1_dec = cortex._regions["S1"].dendritic_decoder
    s2_dec = DendriticDecoder(source_dim=s2.n_l23_total, n_segments=16, n_synapses=48)
    s3_dec = DendriticDecoder(source_dim=s3.n_l23_total, n_segments=16, n_synapses=48)
    s1_bpc = BPCProbe()
    s2_bpc = BPCProbe()
    s3_bpc = BPCProbe()
    s1_words = WordSelectivityProbe(s1.n_columns)
    s2_words = WordSelectivityProbe(s2.n_columns)
    s3_words = WordSelectivityProbe(s3.n_columns)

    # Capture states
    snaps = {"S1": [], "S2": [], "S3": []}
    for name, region in [("S1", s1), ("S2", s2), ("S3", s3)]:
        orig = region.process

        def make_capture(n, r, o):
            def cap(enc):
                result = o(enc)
                snaps[n].append((r.active_columns.copy(), r.active_l23.copy()))
                return result
            return cap

        region.process = make_capture(name, region, orig)

    # Run
    print(f"Running {len(tokens)} tokens through S1→S2→S3+M1...")
    with contextlib.redirect_stdout(io.StringIO()):
        cortex.run(tokens, log_interval=99999)
    print(f"Captured {len(snaps['S1'])} S1, {len(snaps['S2'])} S2, {len(snaps['S3'])} S3 steps.")

    # Restore
    # (not needed since we're done processing)

    # Analyze
    content = [(t, s) for t, s in tokens if t != EOM_TOKEN and t != STORY_BOUNDARY]
    prev = {
        "S1": np.zeros(s1.n_l23_total, dtype=np.bool_),
        "S2": np.zeros(s2.n_l23_total, dtype=np.bool_),
        "S3": np.zeros(s3.n_l23_total, dtype=np.bool_),
    }

    n = min(len(content), len(snaps["S1"]), len(snaps["S2"]), len(snaps["S3"]))
    print(f"Analyzing {n} content steps...")

    probes = {
        "S1": (s1_bpc, s1_dec, s1_words),
        "S2": (s2_bpc, s2_dec, s2_words),
        "S3": (s3_bpc, s3_dec, s3_words),
    }

    for i in range(n):
        tid, tstr = content[i]
        for name, (bpc, dec, words) in probes.items():
            if i > 0:
                bpc.step(tid, prev[name], dec)
            dec.observe(tid, prev[name])
            words.step(tstr, snaps[name][i][0])
            prev[name] = snaps[name][i][1]

    # Results
    print("\n" + "=" * 60)
    print("  CHARACTER-LEVEL PREDICTION (BPC)")
    print("=" * 60)
    for name, (bpc, dec, _) in probes.items():
        print(f"  {name}: {bpc.bpc:.3f} BPC ({dec.n_tokens} tokens)")

    best = min(probes.items(), key=lambda x: x[1][0].bpc)
    print(f"  Best: {best[0]}")

    print("\n" + "=" * 60)
    print("  WORD-LEVEL SELECTIVITY")
    print("=" * 60)
    for name, (_, _, words) in probes.items():
        s = words.summary()
        print(f"\n--- {name} ({s['columns_with_words']} cols) ---")
        print(f"  Consistent words (J>0.3): {s['consistent_words']}")
        print(f"  Mean consistency: {s['mean_consistency']:.3f}")
        print(f"  Selective cols (ent<0.7): {s['selective_columns']}")
        print(f"  Mean selectivity: {s['mean_selectivity']:.3f}")
        if s["top_consistent"]:
            top = ", ".join(f"'{w}'({j:.2f})" for w, j, _ in s["top_consistent"][:5])
            print(f"  Top words: {top}")


if __name__ == "__main__":
    main()
