#!/usr/bin/env python3
"""Full staged pipeline with L5 apical segments enabled on sensory regions."""

from dataclasses import replace

import step.env  # noqa: F401
from step.cortex.canonical import build_canonical_circuit
from step.cortex.stages import BABBLING_STAGE, SENSORY_STAGE
from step.data import inject_eom_tokens, prepare_tokens_charlevel
from step.encoders.positional import PositionalCharEncoder

APICAL_OVERRIDE = {"use_l5_apical_segments": True}


def load_data(n_tokens):
    vocab_sample = max(n_tokens, 1_000_000)
    all_tokens = prepare_tokens_charlevel(vocab_sample, dataset="babylm")
    alphabet = sorted({ch for _, ch in all_tokens if _ >= 0})
    encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
    tokens = all_tokens[:n_tokens] if n_tokens < vocab_sample else all_tokens
    tokens = inject_eom_tokens(tokens, segment_length=200)
    return tokens, encoder


tokens, encoder = load_data(300_000)

cortex = build_canonical_circuit(
    encoder,
    log_interval=5000,
    timeline_interval=0,
    s1_overrides=APICAL_OVERRIDE,
    s2_overrides=APICAL_OVERRIDE,
    s3_overrides=APICAL_OVERRIDE,
    finalize=False,
)
cortex.finalize()

# Import run_stage from cortex_staged for stage execution
from scripts.cortex_staged import run_stage  # noqa: E402

# Stage 1: Sensory (300k)
sensory = replace(
    SENSORY_STAGE,
    n_tokens=300_000,
    save_checkpoint="stage1_sensory_l5apical",
)
run_stage(cortex, sensory, tokens, log_interval=5000)

# Stage 2: Babbling (100k)
babbling = replace(
    BABBLING_STAGE,
    n_tokens=100_000,
    save_checkpoint="stage2_babbling_l5apical",
    load_checkpoint=None,
)
run_stage(cortex, babbling, tokens, log_interval=5000)

print("\nFull pipeline complete.")
