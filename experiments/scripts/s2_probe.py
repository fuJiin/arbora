"""Probe S2 representations: character decoding + word-level selectivity.

Loads a checkpoint and runs corpus through the hierarchy, measuring:
1. S2 dendritic decoder accuracy on next-character prediction (vs S1 baseline)
2. S2 column word-level selectivity (do columns specialize for words?)
3. S1 column word-level selectivity (for comparison)

Runs tokens through topology.run() in one batch for performance,
then analyzes the per-step snapshots captured during the run.
"""

import sys

sys.path.insert(0, "src")

import numpy as np

from step.config import (
    _default_motor_config,
    _default_region2_config,
    _default_s1_config,
    make_motor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import RewardModulator, SurpriseTracker, ThalamicGate
from step.cortex.topology import Topology
from step.data import EOM_TOKEN, STORY_BOUNDARY, prepare_tokens_personachat
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

    m1_cfg = _default_motor_config()
    m1 = make_motor_region(m1_cfg, s1.n_l23_total, seed=456)

    bg = BasalGanglia(s1_cfg.n_columns + 1)

    cortex = Topology(encoder)
    cortex.add_region("S1", s1, entry=True)
    cortex.add_region("S2", s2)
    cortex.add_region("M1", m1, basal_ganglia=bg)

    cortex.connect("S1", "S2", "feedforward", buffer_depth=4, burst_gate=True)
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S1", "M1", "feedforward")
    cortex.connect("S1", "S2", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("M1", "S1", "reward", reward_modulator=RewardModulator())

    return cortex, encoder, s1, s2, m1


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Probe S2 representations")
    parser.add_argument(
        "--checkpoint",
        default="personachat_k4_100k",
        help="Checkpoint name to load",
    )
    parser.add_argument(
        "--chars",
        type=int,
        default=20000,
        help="Number of corpus chars to run through",
    )
    parser.add_argument(
        "--dataset",
        default="personachat",
        choices=["personachat", "babylm"],
        help="Dataset to probe with",
    )
    args = parser.parse_args()

    # Load data — vocab must match checkpoint's training dataset
    if args.dataset == "babylm":
        from step.data import prepare_tokens_charlevel

        print("Loading BabyLM vocabulary + probe data...")
        vocab_tokens = prepare_tokens_charlevel(100000, dataset="babylm")
        alphabet = sorted({ch for _, ch in vocab_tokens if _ >= 0})
        tokens = prepare_tokens_charlevel(args.chars, dataset="babylm")
    else:
        print("Loading PersonaChat vocabulary + probe data...")
        vocab_tokens = prepare_tokens_personachat(100000, speak_window=5)
        alphabet = sorted({ch for _, ch in vocab_tokens if _ >= 0})
        tokens = prepare_tokens_personachat(args.chars, speak_window=5)

    # Build model and load checkpoint
    cortex, _encoder, s1, s2, _m1 = build_model(alphabet)
    ckpt_path = f"experiments/checkpoints/{args.checkpoint}.ckpt"
    print(f"Loading checkpoint: {ckpt_path}")
    cortex.load_checkpoint(ckpt_path)

    # Create probes
    s2_decoder = DendriticDecoder(
        source_dim=s2.n_l23_total,
        n_segments=16,
        n_synapses=48,
    )
    s1_bpc = BPCProbe()
    s2_bpc = BPCProbe()
    s1_words = WordSelectivityProbe(s1.n_columns)
    s2_words = WordSelectivityProbe(s2.n_columns)

    # Monkey-patch: capture per-step S2 state inside topology.run()
    # We'll collect snapshots by wrapping S2's process method.
    snapshots = []
    original_s2_process = s2.process

    def capturing_process(encoding):
        result = original_s2_process(encoding)
        snapshots.append(
            {
                "active_columns": s2.active_columns.copy(),
                "l23": s2.active_l23.copy(),
            }
        )
        return result

    s2.process = capturing_process

    # Also capture S1 state
    s1_snapshots = []
    original_s1_process = s1.process

    def capturing_s1_process(encoding):
        result = original_s1_process(encoding)
        s1_snapshots.append(
            {
                "active_columns": s1.active_columns.copy(),
                "l23": s1.active_l23.copy(),
            }
        )
        return result

    s1.process = capturing_s1_process

    # Run all tokens through hierarchy in one batch
    print(f"Running {len(tokens)} tokens through hierarchy...")
    cortex.run(tokens, log_interval=5000)
    print(f"Done. Captured {len(s1_snapshots)} S1 steps, {len(snapshots)} S2 steps.")

    # Restore original methods
    s2.process = original_s2_process
    s1.process = original_s1_process

    # Build token list excluding EOM/STORY_BOUNDARY
    content_tokens = [
        (tid, tstr)
        for tid, tstr in tokens
        if tid != EOM_TOKEN and tid != STORY_BOUNDARY
    ]

    # Analyze snapshots
    s1_decoder = cortex._regions["S1"].dendritic_decoder
    prev_s1_l23 = np.zeros(s1.n_l23_total, dtype=np.bool_)
    prev_s2_l23 = np.zeros(s2.n_l23_total, dtype=np.bool_)

    n_steps = min(len(content_tokens), len(s1_snapshots), len(snapshots))
    print(f"Analyzing {n_steps} content steps...")

    for i in range(n_steps):
        token_id, token_str = content_tokens[i]

        # BPC: measure prediction from previous state
        if i > 0:
            s1_bpc.step(token_id, prev_s1_l23, s1_decoder)
            s2_bpc.step(token_id, prev_s2_l23, s2_decoder)

        # Train S2 decoder: previous S2 L2/3 → current token
        s2_decoder.observe(token_id, prev_s2_l23)

        # Word selectivity
        s1_words.step(token_str, s1_snapshots[i]["active_columns"])
        s2_words.step(token_str, snapshots[i]["active_columns"])

        # Save for next step
        prev_s1_l23 = s1_snapshots[i]["l23"]
        prev_s2_l23 = snapshots[i]["l23"]

    # === Results ===
    print("\n" + "=" * 60)
    print("  CHARACTER-LEVEL PREDICTION (BPC)")
    print("=" * 60)
    print(f"  S1 dendritic decoder: {s1_bpc.bpc:.3f} BPC")
    print(f"  S2 dendritic decoder: {s2_bpc.bpc:.3f} BPC")
    print(f"  S2 decoder tokens learned: {s2_decoder.n_tokens}")
    if s2_bpc.bpc < s1_bpc.bpc:
        print(f"  S2 BETTER by {s1_bpc.bpc - s2_bpc.bpc:.3f} BPC")
    else:
        print(f"  S1 better by {s2_bpc.bpc - s1_bpc.bpc:.3f} BPC")

    print("\n" + "=" * 60)
    print("  WORD-LEVEL SELECTIVITY")
    print("=" * 60)

    for name, probe in [("S1", s1_words), ("S2", s2_words)]:
        s = probe.summary()
        print(f"\n--- {name} ---")
        print(f"  Words observed: {s['total_words']} ({s['unique_words']} unique)")
        print(f"  Columns with word data: {s['columns_with_words']}")
        print(f"  Selective columns (entropy < 0.7): {s['selective_columns']}")
        print(f"  Mean selectivity (0=perfect, 1=uniform): {s['mean_selectivity']:.3f}")
        print(f"  Consistent words (Jaccard > 0.3): {s['consistent_words']}")
        print(f"  Mean word consistency: {s['mean_consistency']:.3f}")

        if s["top_selective"]:
            print("\n  Most selective columns:")
            for col, ent, word in s["top_selective"]:
                print(f"    col {col:3d}: entropy={ent:.3f}, best word='{word}'")

        if s["top_consistent"]:
            print("\n  Most consistent words:")
            for word, jacc, n in s["top_consistent"][:10]:
                print(f"    '{word}': Jaccard={jacc:.3f} ({n} occurrences)")


if __name__ == "__main__":
    main()
