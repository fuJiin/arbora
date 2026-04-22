"""Text-modality exploration: T1 next-character prediction.

ARB-129 epic. Goal is to form intuition about sparse-encoding
representation quality by building and probing a minimal char-level
circuit before layering more machinery on top.

This package is a substrate, not a product:
- `data` — dictionary loader + train/test split + char-stream iterator.
- `trainer` — `T1Trainer` that drives a region one char at a time with
  caller-controlled word-boundary resets.
- `train` — CLI entry point that wires T1 + decoder + BPC for a single
  end-to-end run.
"""
