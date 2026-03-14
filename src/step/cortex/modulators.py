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


class RewardModulator:
    """Dopaminergic reward signal: third factor in three-factor learning rule.

    Models VTA/SNc dopamine projections to cortex. Reward gates eligibility
    trace consolidation: dw ~ pre x post x reward.

    Tracks reward relative to a slowly-adapting baseline (like SurpriseTracker).
    Positive surprise → modulator > 1 (reinforce). Negative → modulator < 1
    (weaken). Clipped to [0, 2] range.

    Unlike surprise (which is always-on from burst rates), reward is an
    externally-provided signal computed from task performance.
    """

    def __init__(
        self,
        baseline_decay: float = 0.99,
        ema_decay: float = 0.95,
    ):
        self.baseline_decay = baseline_decay
        self._ema_decay = ema_decay
        self._reward_ema: float = 0.0
        self._baseline: float = 0.0

    def update(self, reward: float) -> float:
        """Update with current reward, return modulator (0-2 range).

        reward > 0: positive outcome (reinforce recent learning)
        reward < 0: negative outcome (weaken recent learning)
        reward = 0: neutral (modulator ≈ 1.0)
        """
        self._reward_ema = (
            self._ema_decay * self._reward_ema
            + (1.0 - self._ema_decay) * reward
        )
        self._baseline = (
            self.baseline_decay * self._baseline
            + (1.0 - self.baseline_decay) * reward
        )
        # Modulator centered at 1.0: positive reward → >1, negative → <1
        modulator = 1.0 + (self._reward_ema - self._baseline)
        return max(0.0, min(modulator, 2.0))

    def reset(self) -> None:
        self._reward_ema = 0.0
        self._baseline = 0.0

    @property
    def value(self) -> float:
        """Current modulator value without updating."""
        mod = 1.0 + (self._reward_ema - self._baseline)
        return max(0.0, min(mod, 2.0))


class ThalamicGate:
    """Receiver-side gating: suppresses feedback when receiver is still learning.

    Models the pulvinar modulated by L6 projections from receiving cortex.
    readiness = 1.0 - smoothed_burst_rate
    """

    def __init__(self, ema_decay: float = 0.95):
        self._ema_decay = ema_decay
        self._burst_ema: float = 1.0  # Start closed

    @property
    def readiness(self) -> float:
        return 1.0 - self._burst_ema

    def update(self, burst_rate: float) -> float:
        self._burst_ema = (
            self._ema_decay * self._burst_ema
            + (1.0 - self._ema_decay) * burst_rate
        )
        return self.readiness

    def reset(self) -> None:
        self._burst_ema = 1.0
