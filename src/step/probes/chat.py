"""Chat-specific probes for dialogue environments.

ChatLaminaProbe — stimulus-labeled L2/3 metrics (linear probe, ctx disc).
ChatMotorProbe — motor accuracy, BG gating, turn-taking counters.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from step.cortex.motor import MotorRegion
from step.probes.core import LaminaProbe
from step.probes.snapshots import LaminaRegionSnapshot, MotorRegionSnapshot

if TYPE_CHECKING:
    from step.cortex.circuit import Circuit


class ChatLaminaProbe(LaminaProbe):
    """Extends LaminaProbe with chat-specific L2/3 KPIs.

    Requires stimulus_id kwarg in observe() for:
    - Linear probe accuracy (SGDClassifier on frozen L2/3 -> token)
    - Context discrimination (Jaccard distance across contexts)
    """

    name: str = "chat_lamina"

    def __init__(
        self,
        *,
        l23_sample_interval: int = 10,
        linear_probe_fit_interval: int = 5000,
        linear_probe_window: int = 3000,
        ctx_disc_min_contexts: int = 3,
    ):
        super().__init__(l23_sample_interval=l23_sample_interval)
        self._fit_interval = linear_probe_fit_interval
        self._probe_window = linear_probe_window
        self._ctx_min = ctx_disc_min_contexts

        # Per-region linear probe state
        self._probe_X: dict[str, list[np.ndarray]] = defaultdict(list)
        self._probe_y: dict[str, list[int]] = defaultdict(list)
        self._probe_accuracy: dict[str, float] = {}

        # Per-region context discrimination state
        self._prev_token: int | None = None
        self._bigram_patterns: dict[
            str, dict[tuple[int, int], list[frozenset[int]]]
        ] = defaultdict(lambda: defaultdict(list))
        self._token_contexts: dict[str, dict[int, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Observe with optional stimulus_id for chat-specific KPIs."""
        super().observe(circuit, **kwargs)

        stimulus_id = kwargs.get("stimulus_id")
        if stimulus_id is None:
            return

        for region_name, state in circuit._regions.items():
            region = state.region
            if region.n_l23 == 0:
                continue

            l23_active = region.l23.active
            self._observe_linear_probe(
                region_name,
                l23_active,
                stimulus_id,
            )
            self._observe_ctx_disc(region_name, l23_active, stimulus_id)

        self._prev_token = stimulus_id

        # Periodic linear probe fit
        if self._step_count % self._fit_interval == 0:
            self._fit_linear_probes()

    def boundary(self) -> None:
        """Reset per-dialogue state."""
        self._prev_token = None

    def snapshot(self) -> dict[str, LaminaRegionSnapshot]:
        """Extend parent snapshot with chat-specific L2/3 KPIs."""
        result = super().snapshot()

        # Ensure we have a recent fit
        self._fit_linear_probes()

        for region_name, region_snap in result.items():
            region_snap.l23.linear_probe = self._probe_accuracy.get(region_name, 0.0)
            region_snap.l23.ctx_disc = self._compute_ctx_disc(region_name)

        return result

    # -----------------------------------------------------------------------
    # Per-region observe helpers
    # -----------------------------------------------------------------------

    def _observe_linear_probe(
        self,
        region_name: str,
        l23_active: np.ndarray,
        stimulus_id: int,
    ) -> None:
        """Accumulate (activation, label) pairs for linear probe."""
        self._probe_X[region_name].append(l23_active.astype(np.float32))
        self._probe_y[region_name].append(stimulus_id)
        # Cap buffer
        window = self._probe_window * 2
        if len(self._probe_X[region_name]) > window:
            self._probe_X[region_name] = self._probe_X[region_name][
                -self._probe_window :
            ]
            self._probe_y[region_name] = self._probe_y[region_name][
                -self._probe_window :
            ]

    def _observe_ctx_disc(
        self,
        region_name: str,
        l23_active: np.ndarray,
        stimulus_id: int,
    ) -> None:
        """Track bigram context patterns for discrimination metric."""
        if self._prev_token is not None:
            neurons = frozenset(int(n) for n in np.nonzero(l23_active)[0])
            key = (self._prev_token, stimulus_id)
            self._bigram_patterns[region_name][key].append(neurons)
            self._token_contexts[region_name][stimulus_id].add(self._prev_token)

    # -----------------------------------------------------------------------
    # Fit / compute
    # -----------------------------------------------------------------------

    def _fit_linear_probes(self) -> None:
        """Fit SGD linear classifiers on accumulated L2/3 data."""
        try:
            from sklearn.linear_model import SGDClassifier
        except ImportError:
            return

        for region_name in list(self._probe_X.keys()):
            X_list = self._probe_X[region_name]
            y_list = self._probe_y[region_name]
            if len(X_list) < 100:
                continue

            X = np.array(X_list[-self._probe_window :])
            y = np.array(y_list[-self._probe_window :])

            # 80/20 split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            if len(set(y_test)) < 2:
                continue

            clf = SGDClassifier(
                loss="modified_huber",
                max_iter=100,
                random_state=42,
                n_jobs=1,
            )
            clf.fit(X_train, y_train)
            self._probe_accuracy[region_name] = float(clf.score(X_test, y_test))

    def _compute_ctx_disc(self, region_name: str) -> float:
        """Mean Jaccard distance for same token across different contexts."""
        bigrams = self._bigram_patterns.get(region_name, {})
        contexts = self._token_contexts.get(region_name, {})
        if not bigrams:
            return 0.0

        rng = np.random.default_rng(42)
        all_dists: list[float] = []

        for tid, ctx_set in contexts.items():
            if len(ctx_set) < self._ctx_min:
                continue
            # Collect all patterns for this token across contexts
            patterns: list[frozenset[int]] = []
            for (_prev, t), pats in bigrams.items():
                if t == tid:
                    patterns.extend(pats)
            if len(patterns) < self._ctx_min:
                continue

            # Sample pairwise Jaccard distances
            n = len(patterns)
            pairs = min(50, n * (n - 1) // 2)
            for _ in range(pairs):
                i, j = rng.choice(n, 2, replace=False)
                s1, s2 = patterns[i], patterns[j]
                union = len(s1 | s2)
                if union > 0:
                    all_dists.append(1.0 - len(s1 & s2) / union)

        return float(np.mean(all_dists)) if all_dists else 0.0


# ---------------------------------------------------------------------------
# ChatMotorProbe
# ---------------------------------------------------------------------------


class ChatMotorProbe:
    """Per-region motor metrics accumulator for chat environments.

    Tracks motor output accuracy, BG gating, turn-taking behavior,
    and reward signals. Reads motor region state after each process().

    Chat-prefixed because turn-taking (EOM/input phases) is
    dialogue-specific. Does NOT train motor decoders.
    """

    name: str = "motor"
    MAX_SPEAK_STEPS: int = 20

    def __init__(self):
        # Per-region accumulators
        self._motor_accuracies: dict[str, list[float]] = defaultdict(list)
        self._motor_confidences: dict[str, list[float]] = defaultdict(list)
        self._motor_rewards: dict[str, list[float]] = defaultdict(list)
        self._bg_gate_values: dict[str, list[float]] = defaultdict(list)
        # Turn-taking counters
        self._turn_eom_steps: dict[str, int] = defaultdict(int)
        self._turn_input_steps: dict[str, int] = defaultdict(int)
        self._turn_correct_speak: dict[str, int] = defaultdict(int)
        self._turn_correct_silent: dict[str, int] = defaultdict(int)
        self._turn_interruptions: dict[str, int] = defaultdict(int)
        self._turn_unresponsive: dict[str, int] = defaultdict(int)
        self._turn_rambles: dict[str, int] = defaultdict(int)

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Read motor state from circuit after process().

        Expects kwargs: stimulus_id, in_eom, eom_steps (from train loop).
        """
        stimulus_id = kwargs.get("stimulus_id")
        in_eom = kwargs.get("in_eom", False)
        eom_steps = kwargs.get("eom_steps", 0)

        for name, s in circuit._regions.items():
            if not s.motor:
                continue
            if not isinstance(s.region, MotorRegion):
                continue
            motor = s.region

            if circuit._total_steps == 0:
                continue

            # BG gate value
            if s.basal_ganglia is not None:
                self._bg_gate_values[name].append(motor.last_gate)

            # Motor output
            m_id, m_conf = motor.last_output
            self._motor_confidences[name].append(m_conf)

            if m_id >= 0 and stimulus_id is not None:
                self._motor_accuracies[name].append(1.0 if m_id == stimulus_id else 0.0)

            # Reward
            self._motor_rewards[name].append(motor.last_reward)

            # Turn-taking (reads EOM state from kwargs, not circuit)
            spoke = m_id >= 0
            if in_eom:
                self._turn_eom_steps[name] += 1
                if spoke:
                    if eom_steps > self.MAX_SPEAK_STEPS:
                        self._turn_rambles[name] += 1
                    else:
                        self._turn_correct_speak[name] += 1
                else:
                    self._turn_unresponsive[name] += 1
            else:
                self._turn_input_steps[name] += 1
                if spoke:
                    self._turn_interruptions[name] += 1
                else:
                    self._turn_correct_silent[name] += 1

    def snapshot(self) -> dict[str, MotorRegionSnapshot]:
        """Return per-region motor metrics."""
        all_regions = set(
            list(self._motor_confidences.keys()) + list(self._bg_gate_values.keys())
        )
        result: dict[str, MotorRegionSnapshot] = {}
        for name in sorted(all_regions):
            result[name] = MotorRegionSnapshot(
                motor_accuracies=self._motor_accuracies.get(name, []),
                motor_confidences=self._motor_confidences.get(name, []),
                motor_rewards=self._motor_rewards.get(name, []),
                bg_gate_values=self._bg_gate_values.get(name, []),
                turn_eom_steps=self._turn_eom_steps.get(name, 0),
                turn_input_steps=self._turn_input_steps.get(name, 0),
                turn_correct_speak=self._turn_correct_speak.get(name, 0),
                turn_correct_silent=self._turn_correct_silent.get(name, 0),
                turn_interruptions=self._turn_interruptions.get(name, 0),
                turn_unresponsive=self._turn_unresponsive.get(name, 0),
                turn_rambles=self._turn_rambles.get(name, 0),
            )
        return result
