"""Surprise-modulated learning signal from Region 1 burst rate.

Models third-factor neuromodulation: norepinephrine from locus coeruleus
scales plasticity based on surprise (deviation from baseline burst rate).
Higher regions learn more when lower regions are surprised.
"""


class SurpriseTracker:
    """Track burst rate relative to baseline, output a learning modulator.

    The modulator is the ratio of smoothed burst rate to a slowly-adapting
    baseline. When burst rate spikes above baseline, modulator > 1 (more
    learning). At steady state, modulator ≈ 1.

    O(1) per step, negligible memory.
    """

    def __init__(
        self,
        baseline_decay: float = 0.99,
        min_baseline: float = 0.01,
        ema_decay: float = 0.95,
    ):
        self.baseline_decay = baseline_decay
        self.min_baseline = min_baseline
        self._ema_decay = ema_decay
        self.baseline_burst_rate: float = 0.5
        self._burst_ema: float = 0.5

    def update(self, burst_rate: float) -> float:
        """Update with current burst rate, return surprise modulator (0-2 range)."""
        self._burst_ema = (
            self._ema_decay * self._burst_ema + (1 - self._ema_decay) * burst_rate
        )
        self.baseline_burst_rate = (
            self.baseline_decay * self.baseline_burst_rate
            + (1 - self.baseline_decay) * burst_rate
        )
        baseline = max(self.baseline_burst_rate, self.min_baseline)
        surprise = self._burst_ema / baseline
        return min(surprise, 2.0)
