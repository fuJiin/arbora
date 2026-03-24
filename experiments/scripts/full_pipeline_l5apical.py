#!/usr/bin/env python3
"""Full staged pipeline with L5 apical segments enabled on sensory regions."""

import os
from dataclasses import replace

import step.config as cfg
import step.env  # noqa: F401
from step.cortex.stages import BABBLING_STAGE, SENSORY_STAGE

# Patch defaults to enable L5 apical segments on sensory regions
_orig_s1 = cfg._default_s1_config
_orig_r2 = cfg._default_region2_config
_orig_r3 = cfg._default_region3_config

cfg._default_s1_config = lambda: replace(_orig_s1(), use_l5_apical_segments=True)
cfg._default_region2_config = lambda: replace(_orig_r2(), use_l5_apical_segments=True)
cfg._default_region3_config = lambda: replace(_orig_r3(), use_l5_apical_segments=True)

import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.cortex_staged import build_topology, load_data, run_stage  # noqa: E402

tokens, encoder = load_data(300_000)

cortex = build_topology(encoder, log_interval=5000)
cortex.finalize()

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
