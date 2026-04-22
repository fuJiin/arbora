"""T1Trainer — drives a region through a char stream with word resets.

Responsibilities:
- Own the region, encoder, decoder, and BPC probe for a single run.
- Per step: use the L2/3 state *before* the current char to predict it,
  update BPC/accuracy, observe decoder, process char through region.
- `reset()` clears region state and the stored pre-step L2/3 so the
  caller can mark word boundaries without in-band sentinel chars.

Caller controls word boundaries (loop over words, `reset()` per word).

Eval mode (`train=False`) disables decoder learning and freezes the
region's own learning via `region.learning_enabled`. BPC still
accumulates so the caller can read held-out predictive quality off
the same probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arbora.cortex.sensory import SensoryRegion
from arbora.decoders.dendritic import DendriticDecoder
from arbora.encoders.charbit import CharbitEncoder
from arbora.probes.bpc import BPCProbe


@dataclass
class StepResult:
    """Info for one step of the trainer, returned by `step()`.

    Useful for the explorer notebook (PR C) and for diagnostic probes
    (PR B) to attach to without re-running the trainer loop.
    """

    char: str
    token_id: int
    bits: float
    top1_char: str | None
    top1_correct: bool


class T1Trainer:
    """Drive a T1 region through a character stream.

    Parameters
    ----------
    region : SensoryRegion
        The T1 region. Typically built from `_default_t1_config`
        with `input_dim=encoder.input_dim`.
    encoder : CharbitEncoder
        Shared with the decoder's alphabet (the decoder indexes by
        `ord(char)`, so any char in `encoder` is a valid token).
    decoder : DendriticDecoder
        Dendritic segment decoder over L2/3 → next-char distribution.
    bpc_probe : BPCProbe, optional
        If provided, accumulates per-step BPC. If None, creates one.
    """

    def __init__(
        self,
        region: SensoryRegion,
        encoder: CharbitEncoder,
        decoder: DendriticDecoder,
        bpc_probe: BPCProbe | None = None,
    ):
        self.region = region
        self.encoder = encoder
        self.decoder = decoder
        self.bpc_probe = bpc_probe if bpc_probe is not None else BPCProbe()
        self._prev_l23: np.ndarray = np.zeros(region.n_l23_total, dtype=np.bool_)

    def reset(self) -> None:
        """Clear region state and pre-step L2/3 (call at word boundaries).

        Uses `reset_working_memory()` — wipes activations/firing rates,
        preserves learned weights and segments.
        """
        self.region.reset_working_memory()
        self._prev_l23 = np.zeros(self.region.n_l23_total, dtype=np.bool_)

    def step(self, char: str, *, train: bool = True) -> StepResult:
        """Process one character.

        Pipeline:
          1. Predict `char` from pre-step L2/3 (the state from *before*
             this char arrived).
          2. Record bits + top-1 match.
          3. If `train`, decoder.observe(char, pre-step L2/3) — learns
             that this L2/3 pattern predicts `char`.
          4. Process the encoded char through the region. `train`
             toggles `region.learning_enabled` for the duration.
          5. Update stored pre-step L2/3 to the post-step state.
        """
        token_id = ord(char)

        bits = self.bpc_probe.step(token_id, self._prev_l23, self.decoder)
        top1_char, top1_correct = self._top1(token_id)

        if train:
            self.decoder.observe(token_id, self._prev_l23)

        encoding = self.encoder.encode(char).flatten()
        prior = self.region.learning_enabled
        self.region.learning_enabled = train
        try:
            self.region.process(encoding)
        finally:
            self.region.learning_enabled = prior

        self._prev_l23 = self.region.l23.active.copy()

        return StepResult(
            char=char,
            token_id=token_id,
            bits=bits,
            top1_char=top1_char,
            top1_correct=top1_correct,
        )

    def train_word(self, word: str, *, train: bool = True) -> list[StepResult]:
        """Reset then step through every char in `word`."""
        self.reset()
        return [self.step(c, train=train) for c in word]

    def _top1(self, actual_token_id: int) -> tuple[str | None, bool]:
        scores = self.decoder.decode_scores(self._prev_l23)
        if not scores:
            return None, False
        top_token = max(scores, key=lambda k: scores[k])
        return chr(top_token), top_token == actual_token_id
