"""T1Trainer — drives a region through a char stream with word resets.

Responsibilities (minimal by design):
- Own the region + encoder for a single run.
- Per step: encode the char, toggle region learning on/off, call
  `region.process`, return a `StepResult` with the post-step L2/3
  active pattern for observation.
- `reset()` clears the region's working memory so the caller can mark
  word boundaries without in-band sentinel chars.

**Observation-first.** The decoder + BPC probe that the CLI uses for
prediction metrics are *optional* — pass them in to get per-step bits
and top-1 accuracy on `StepResult`, or omit them to just watch L2/3
evolve. The trainer's core purpose is to drive the region; decoding is
a layer on top.

The encoder argument is duck-typed: anything with an `.encode(str) ->
np.ndarray` method and an `.input_dim` property works. Both
`arbora.encoders.charbit.CharbitEncoder` (multi-char) and
`arbora.encoders.onehot.OneHotCharEncoder` (single-char) are valid —
this module flattens multi-dim encodings to 1D before feeding the
region.

Eval mode (`train=False`) freezes `region.learning_enabled` for the
duration of the step and, if a decoder is wired in, skips its
`observe()` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from arbora.cortex.sensory import SensoryRegion
    from arbora.decoders.dendritic import DendriticDecoder
    from arbora.probes.bpc import BPCProbe


@dataclass
class StepResult:
    """Info for one step of the trainer.

    `bits`, `top1_char`, `top1_correct` are only populated if a decoder
    (and BPC probe) were passed to the trainer. Otherwise they're
    None/False — the trainer is running in pure-observation mode.
    """

    char: str
    token_id: int
    l23_active: np.ndarray
    bits: float | None = None
    top1_char: str | None = None
    top1_correct: bool = False


class T1Trainer:
    """Drive a T1 region through a character stream.

    Parameters
    ----------
    region : SensoryRegion
        The T1 region. Typically built from `_default_t1_config`
        with `input_dim=encoder.input_dim`.
    encoder : object with `encode(char) -> np.ndarray` and `input_dim`
        Any char-level encoder. Multi-char-array outputs are flattened.
    decoder : DendriticDecoder, optional
        If provided, `step()` will use the pre-step L2/3 to predict the
        current char via the decoder, and (when training) observe it.
        If None, no prediction pipeline runs.
    bpc_probe : BPCProbe, optional
        Required for populating `StepResult.bits`. Only used if
        `decoder` is also provided. If None, bits stays None.
    """

    def __init__(
        self,
        region: SensoryRegion,
        encoder: Any,
        decoder: DendriticDecoder | None = None,
        bpc_probe: BPCProbe | None = None,
    ):
        self.region = region
        self.encoder = encoder
        self.decoder = decoder
        self.bpc_probe = bpc_probe
        self._prev_l23: np.ndarray = np.zeros(region.n_l23_total, dtype=np.bool_)

    def reset(self) -> None:
        """Clear the region's working memory and the pre-step L2/3 buffer.

        Uses `reset_working_memory()` — wipes activations/firing rates,
        preserves learned weights and segments. Call at word boundaries.
        """
        self.region.reset_working_memory()
        self._prev_l23 = np.zeros(self.region.n_l23_total, dtype=np.bool_)

    def step(self, char: str, *, train: bool = True) -> StepResult:
        """Process one character.

        Pipeline:
          1. If decoder + bpc_probe are wired, use pre-step L2/3 to
             predict `char`: record bits + top-1 match.
          2. If `train` and decoder is wired: `decoder.observe(char,
             pre-step L2/3)` — learns that this L2/3 pattern precedes
             `char`.
          3. Toggle `region.learning_enabled = train`, process encoded
             char, restore prior flag.
          4. Update stored pre-step L2/3 to the post-step active set.
        """
        token_id = ord(char)

        bits: float | None = None
        top1_char: str | None = None
        top1_correct = False
        if self.decoder is not None:
            if self.bpc_probe is not None:
                bits = self.bpc_probe.step(token_id, self._prev_l23, self.decoder)
            top1_char, top1_correct = self._top1(token_id)
            if train:
                self.decoder.observe(token_id, self._prev_l23)

        encoding = self.encoder.encode(char)
        if encoding.ndim > 1:
            encoding = encoding.flatten()

        prior = self.region.learning_enabled
        self.region.learning_enabled = train
        try:
            self.region.process(encoding)
        finally:
            self.region.learning_enabled = prior

        active = self.region.l23.active.copy()
        self._prev_l23 = active

        return StepResult(
            char=char,
            token_id=token_id,
            l23_active=active,
            bits=bits,
            top1_char=top1_char,
            top1_correct=top1_correct,
        )

    def train_word(self, word: str, *, train: bool = True) -> list[StepResult]:
        """Reset then step through every char in `word`."""
        self.reset()
        return [self.step(c, train=train) for c in word]

    def _top1(self, actual_token_id: int) -> tuple[str | None, bool]:
        if self.decoder is None:
            return None, False
        scores = self.decoder.decode_scores(self._prev_l23)
        if not scores:
            return None, False
        top_token = max(scores, key=lambda k: scores[k])
        return chr(top_token), top_token == actual_token_id
