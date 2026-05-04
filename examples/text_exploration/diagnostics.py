"""Three diagnostic checkpoints for the T1 next-char circuit (ARB-131).

1. `character_sdr_overlap` — Jaccard similarity between L2/3 SDRs for
   each character, comparing within-group vs. across-group (default
   groups: vowels vs. consonants). Answers "does learning cluster
   phonetically-similar chars?"
2. `context_sensitivity` — top-K next-char predictions for different
   prefixes (e.g., `ca_` vs. `de_`). Low top-K overlap = high context
   sensitivity.
3. `weight_distribution` — histograms and pathology flags for
   `ff_weights` and segment permanences. Detects saturation (too many
   at clip ceiling) or collapse (too many near zero).

All three are non-destructive: they either re-run the region with
`learning_enabled=False` or only read state, never modify it.

Out of scope: the explorer notebook (PR C) will render these. This
module is just the measurement primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from examples.text_exploration.trainer import T1Trainer

# Phonetic-partition analysis alphabet: only a-z, no space. Space
# isn't phonetic, so it doesn't belong in the vowel/consonant split.
# `character_sdr_overlap` uses this by default; callers can pass any
# alphabet string they want.
PHONETIC_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOWELS = "aeiou"
CONSONANTS = "".join(c for c in PHONETIC_ALPHABET if c not in VOWELS)


# ---------------------------------------------------------------------------
# Checkpoint 1: SDR overlap sanity
# ---------------------------------------------------------------------------


@dataclass
class LaminaOverlapStats:
    """Pairwise Jaccard stats for one lamina's per-char SDRs.

    Collected as: for each char, `reset + step(char) + read lamina.active`.
    Pairwise Jaccard is partitioned into vowel-vowel, consonant-
    consonant, and across-group pairs.
    """

    per_char_sdr: dict[str, np.ndarray]
    within_vowel: list[float]
    within_consonant: list[float]
    across: list[float]

    @property
    def within_vowel_mean(self) -> float:
        return float(np.mean(self.within_vowel)) if self.within_vowel else 0.0

    @property
    def within_consonant_mean(self) -> float:
        return float(np.mean(self.within_consonant)) if self.within_consonant else 0.0

    @property
    def across_mean(self) -> float:
        return float(np.mean(self.across)) if self.across else 0.0

    @property
    def clustered(self) -> bool:
        """True if phonetic groups show above-across similarity."""
        a = self.across_mean
        return self.within_vowel_mean > a and self.within_consonant_mean > a


@dataclass
class SDROverlapResult:
    """L4 + L2/3 per-char SDR overlap stats.

    Both are captured with one pass (reset + 1 step per char). L4 is
    the input representation driven by `ff_weights`; L2/3 is L4
    projected through `l4_to_l23_weights` (no lateral context after
    reset). Comparing the two disambiguates:

    - L4 collapsed, L2/3 collapsed → ff_weights don't differentiate
      chars at the input layer. Fix the encoder or ff learning rule.
    - L4 distinct, L2/3 collapsed → L4→L2/3 projection smushes
      distinct L4 patterns together. The in-region weights are the
      problem, not the input.
    - L4 distinct, L2/3 distinct → the single-region representation
      is healthy; any downstream failure points elsewhere.
    """

    l4: LaminaOverlapStats
    l23: LaminaOverlapStats


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return float((a & b).sum()) / union


def _partition_pairs(
    per_char: dict[str, np.ndarray],
) -> LaminaOverlapStats:
    within_v: list[float] = []
    within_c: list[float] = []
    across: list[float] = []
    char_list = list(per_char.keys())
    for i, c1 in enumerate(char_list):
        for c2 in char_list[i + 1 :]:
            j = _jaccard(per_char[c1], per_char[c2])
            c1_is_v, c2_is_v = c1 in VOWELS, c2 in VOWELS
            if c1_is_v and c2_is_v:
                within_v.append(j)
            elif not c1_is_v and not c2_is_v:
                within_c.append(j)
            else:
                across.append(j)
    return LaminaOverlapStats(
        per_char_sdr=per_char,
        within_vowel=within_v,
        within_consonant=within_c,
        across=across,
    )


def character_sdr_overlap(
    trainer: T1Trainer,
    chars: str = PHONETIC_ALPHABET,
) -> SDROverlapResult:
    """Jaccard between L4 and L2/3 SDRs for each char.

    Collects both laminae in a single pass — the cost is one extra
    array copy per char. Non-destructive: uses `train=False` on each
    step and restores the trainer's stored `_prev_l23` afterwards.
    """
    saved_prev = trainer._prev_l23.copy()
    per_char_l4: dict[str, np.ndarray] = {}
    per_char_l23: dict[str, np.ndarray] = {}
    try:
        for c in chars:
            trainer.reset()
            trainer.step(c, train=False)
            per_char_l4[c] = trainer.region.l4.active.copy()
            per_char_l23[c] = trainer.region.l23.active.copy()
    finally:
        trainer.reset()
        trainer._prev_l23 = saved_prev

    return SDROverlapResult(
        l4=_partition_pairs(per_char_l4),
        l23=_partition_pairs(per_char_l23),
    )


# ---------------------------------------------------------------------------
# Checkpoint 2: Context sensitivity
# ---------------------------------------------------------------------------


@dataclass
class ContextSensitivityResult:
    """Top-K next-char predictions for each prefix and pairwise overlap.

    High context sensitivity = different prefixes produce different
    top-K sets. Low overlap (close to 0) is ideal; full overlap (1.0)
    means the region ignores prefix context.
    """

    prefixes: list[str]
    top_k_per_prefix: dict[str, list[str]]
    pairwise_overlap: dict[tuple[str, str], float] = field(default_factory=dict)

    @property
    def mean_overlap(self) -> float:
        vals = list(self.pairwise_overlap.values())
        return float(np.mean(vals)) if vals else 0.0


def context_sensitivity(
    trainer: T1Trainer,
    prefixes: list[str],
    *,
    k: int = 3,
) -> ContextSensitivityResult:
    """For each prefix, get top-K decoder predictions; compare across.

    Non-destructive: runs each prefix with `train=False` and resets
    between prefixes. Restores the trainer's prior `_prev_l23`.
    """
    saved_prev = trainer._prev_l23.copy()
    top_k: dict[str, list[str]] = {}
    try:
        for prefix in prefixes:
            trainer.reset()
            for c in prefix:
                trainer.step(c, train=False)
            scores = trainer.decoder.decode_scores(trainer._prev_l23)
            # Sort tokens by score descending; decode to chars.
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            top_k[prefix] = [chr(tok) for tok, _ in ranked[:k]]
    finally:
        trainer.reset()
        trainer._prev_l23 = saved_prev

    pairwise: dict[tuple[str, str], float] = {}
    for i, p1 in enumerate(prefixes):
        for p2 in prefixes[i + 1 :]:
            set1, set2 = set(top_k[p1]), set(top_k[p2])
            union = set1 | set2
            pairwise[(p1, p2)] = len(set1 & set2) / len(union) if union else 0.0
    return ContextSensitivityResult(
        prefixes=prefixes,
        top_k_per_prefix=top_k,
        pairwise_overlap=pairwise,
    )


# ---------------------------------------------------------------------------
# Checkpoint 3: Weight distribution stability
# ---------------------------------------------------------------------------


@dataclass
class WeightStats:
    """Summary stats for a weight/permanence array."""

    name: str
    n: int
    mean: float
    std: float
    frac_near_zero: float  # frac in [0, 0.05]
    frac_near_one: float  # frac in [0.95, 1.0]
    frac_below_threshold: float  # frac below perm_threshold (for segments)

    @property
    def saturated(self) -> bool:
        """Pathological if >30% of weights pin to the clip ceiling."""
        return self.frac_near_one > 0.30

    @property
    def collapsed(self) -> bool:
        """Pathological if >80% are effectively zero."""
        return self.frac_near_zero > 0.80


@dataclass
class WeightDistributionResult:
    ff: WeightStats
    l4_lat_perm: WeightStats
    l23_seg_perm: WeightStats
    l5_seg_perm: WeightStats | None

    @property
    def any_pathology(self) -> bool:
        arr = [self.ff, self.l4_lat_perm, self.l23_seg_perm]
        if self.l5_seg_perm is not None:
            arr.append(self.l5_seg_perm)
        return any(s.saturated or s.collapsed for s in arr)


def _weight_stats(
    name: str,
    arr: np.ndarray,
    perm_threshold: float = 0.5,
) -> WeightStats:
    flat = arr.reshape(-1)
    n = int(flat.size)
    if n == 0:
        return WeightStats(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return WeightStats(
        name=name,
        n=n,
        mean=float(flat.mean()),
        std=float(flat.std()),
        frac_near_zero=float((flat < 0.05).mean()),
        frac_near_one=float((flat > 0.95).mean()),
        frac_below_threshold=float((flat < perm_threshold).mean()),
    )


def weight_distribution(
    trainer: T1Trainer,
) -> WeightDistributionResult:
    """Summarize ff_weights and segment permanences for the region.

    Only looks at existing arrays; never mutates state.
    """
    region = trainer.region
    perm_threshold = region.perm_threshold
    # ff_mask==False positions are structural zeros — skip them so stats
    # reflect the *learnable* weights, not the masked-off void.
    ff_flat = region.ff_weights[region.ff_mask]
    ff_stats = _weight_stats("ff_weights", ff_flat, perm_threshold=perm_threshold)

    l4_stats = _weight_stats(
        "l4_lat_seg_perm", region.l4_lat_seg_perm, perm_threshold=perm_threshold
    )
    l23_stats = _weight_stats(
        "l23_seg_perm", region.l23_seg_perm, perm_threshold=perm_threshold
    )
    l5_stats: WeightStats | None = None
    if region.l5_seg_perm.size > 0:
        l5_stats = _weight_stats(
            "l5_seg_perm", region.l5_seg_perm, perm_threshold=perm_threshold
        )
    return WeightDistributionResult(
        ff=ff_stats,
        l4_lat_perm=l4_stats,
        l23_seg_perm=l23_stats,
        l5_seg_perm=l5_stats,
    )


# ---------------------------------------------------------------------------
# Formatted summary
# ---------------------------------------------------------------------------


def format_diagnostics(
    sdr: SDROverlapResult,
    ctx: ContextSensitivityResult,
    weights: WeightDistributionResult,
) -> str:
    lines = []
    lines.append("=== Checkpoint 1: SDR overlap per lamina (reset + 1 char) ===")
    for label, stats in (("L4", sdr.l4), ("L2/3", sdr.l23)):
        lines.append(f"  {label}:")
        lines.append(
            f"    within-vowel    mean Jaccard = {stats.within_vowel_mean:.3f} "
            f"(n={len(stats.within_vowel)})"
        )
        lines.append(
            f"    within-conson.  mean Jaccard = {stats.within_consonant_mean:.3f} "
            f"(n={len(stats.within_consonant)})"
        )
        lines.append(
            f"    across-group    mean Jaccard = {stats.across_mean:.3f} "
            f"(n={len(stats.across)})"
        )
        lines.append(f"    phonetically clustered: {stats.clustered}")
    lines.append("")
    lines.append("=== Checkpoint 2: context sensitivity (top-K next-char) ===")
    for prefix, chars in ctx.top_k_per_prefix.items():
        lines.append(f"  '{prefix}_' → {chars}")
    lines.append(
        f"  mean pairwise top-K overlap = {ctx.mean_overlap:.3f} "
        "(lower = more context-sensitive)"
    )
    lines.append("")
    lines.append("=== Checkpoint 3: weight distribution ===")
    for s in _iter_weight_stats(weights):
        lines.append(
            f"  {s.name}: n={s.n} mean={s.mean:.3f} std={s.std:.3f} "
            f"near_0={s.frac_near_zero:.2f} near_1={s.frac_near_one:.2f} "
            f"<thresh={s.frac_below_threshold:.2f}"
            + (" [SATURATED]" if s.saturated else "")
            + (" [COLLAPSED]" if s.collapsed else "")
        )
    lines.append(f"  any pathology detected: {weights.any_pathology}")
    return "\n".join(lines)


def _iter_weight_stats(w: WeightDistributionResult):
    yield w.ff
    yield w.l4_lat_perm
    yield w.l23_seg_perm
    if w.l5_seg_perm is not None:
        yield w.l5_seg_perm
