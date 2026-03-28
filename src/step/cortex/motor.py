"""Motor cortical region: maps sensory context to output predictions.

Subclasses SensoryRegion (encoding_width=0, full connectivity) and adds
an L5-like output layer with confidence gating. Columns self-organize
to represent output tokens via Hebbian learning — same cortical algorithm
as sensory regions, different wiring.

The column→token mapping is learned: each column tracks which token_id
most frequently activates it, forming a self-organizing motor map.
"""

import numpy as np

from step.config import PlasticityRule
from step.cortex.region import CorticalRegion


class MotorRegion(CorticalRegion):
    """Cortical region with L5 output layer for token production.

    Receives S1's L2/3 firing rate as input (full connectivity, no
    encoding-width structure). Adds:
    - L5 output layer: learned weights mapping L2/3 -> token scores
    - Output weights have structural sparsity (like ff_weights)
    - Three-factor learning: Hebbian + reward modulation on both
      ff_weights (input mapping) and output_weights (L5 readout)
    - Babbling mode: direct column forcing for motor exploration
    """

    def __init__(
        self,
        input_dim: int,
        *,
        n_columns: int = 32,
        n_output_tokens: int = 128,
        output_vocab: list[int] | None = None,
        output_lr: float = 0.05,
        output_sparsity: float = 0.5,
        output_threshold: float = 0.3,
        eligibility_clip: float = 0.05,
        reward_baseline_decay: float = 0.0,
        plasticity_rule: PlasticityRule = PlasticityRule.THREE_FACTOR,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            n_columns=n_columns,
            plasticity_rule=plasticity_rule,
            seed=seed,
            **kwargs,
        )
        self.output_threshold = output_threshold

        # Vocabulary: maps L5 output index -> token_id.
        # If provided, L5 only produces valid tokens.
        # If None, output index IS the token_id (raw ASCII).
        if output_vocab is not None:
            self._output_vocab = np.array(output_vocab, dtype=np.int64)
            n_output_tokens = len(output_vocab)
        else:
            self._output_vocab = None

        # Exploration mode: mix random column forcing with normal feedforward.
        # 0.0 = normal (no noise), 1.0 = pure random exploration.
        # Builds sensorimotor forward model by training ff_weights to map
        # input patterns to randomly forced column activations.
        self.exploration_noise: float = 0.0

        # Eligibility trace clamp: prevents unbounded accumulation over
        # multi-step episodes. 0.0 = no clamping (original behavior).
        self._eligibility_clip = eligibility_clip

        # Reward baseline: running average subtracted before consolidation.
        # When performance is good, baseline rises -> marginal reward -> 0 ->
        # weights stop changing. 0.0 = disabled (original behavior).
        self._reward_baseline_decay = reward_baseline_decay
        self._reward_baseline = 0.0

        # Goal drive scale: slower learning for goal weights (stability).
        self.goal_consolidation_scale: float = 0.3

        # -- L5 output layer: L2/3 -> token scores --
        # Models cortical L5 pyramidal neurons that project to motor
        # periphery. Learned weights with structural sparsity, same
        # architecture as ff_weights. Three-factor Hebbian learning:
        # pre=L2/3 activity, post=token, modulated by reward.
        self.n_output_tokens = n_output_tokens
        self.output_lr = output_lr
        n_l5 = self.n_l5_total

        # L5 output weights: (n_l5_total, n_output_tokens)
        # Maps L5 neuron activity to token scores for motor output.
        self.output_weights = self._rng.uniform(
            0,
            0.01,
            size=(n_l5, n_output_tokens),
        )
        # Structural sparsity: each L5 neuron connects to ~50% of tokens
        self.output_mask = (
            self._rng.random((n_l5, n_output_tokens)) < output_sparsity
        ).astype(np.float64)
        self.output_weights *= self.output_mask

        # L5 eligibility trace (three-factor learning)
        self._output_eligibility = np.zeros((n_l5, n_output_tokens))

        # Last step output (set by circuit run loop after BG gating)
        self.last_output: tuple[int, float] = (-1, 0.0)
        self.last_gate: float = 0.5
        self.last_reward: float = 0.0

    def init_goal_drive(self, source_dim: int) -> None:
        """Initialize PFC->M1 feedforward goal weights.

        Called once when PFC is connected. Creates learned weights that
        map PFC's L2/3 firing rate to additive M1 neuron drive.
        """
        _target_dim = self.input_lamina.n_total
        self._goal_weights = self._rng.uniform(
            0,
            0.01,
            size=(source_dim, _target_dim),
        )
        # Structural sparsity: ~50% connectivity
        goal_mask = self._rng.random((source_dim, _target_dim)) < 0.5
        self._goal_weights *= goal_mask
        self._goal_eligibility = np.zeros_like(self._goal_weights)
        self._goal_drive = None

    def set_goal_drive(self, pfc_firing_rate: np.ndarray) -> None:
        """Set PFC goal signal for next process() call."""
        self._goal_drive = pfc_firing_rate

    def process(
        self,
        encoding: np.ndarray,
        *,
        forced_columns: np.ndarray | None = None,
    ) -> np.ndarray:
        """Feedforward + optional goal drive + L5 output scores.

        Routes to exploration (forced random columns) or normal processing
        (base class handles goal drive injection if set).
        """
        if self.exploration_noise >= 1.0:
            return self._explore_direct(encoding)
        if self.exploration_noise > 0.0 and self._rng.random() < self.exploration_noise:
            return self._explore_direct(encoding)
        return super().process(encoding, forced_columns=forced_columns)

    def _explore_direct(self, encoding: np.ndarray | None = None) -> np.ndarray:
        """Force random column activations while training ff_weights.

        Randomly selects k columns, then delegates to the base class
        process() with forced_columns. The base class handles the full
        step pipeline (predictions, activation, learning, traces).

        ff_weights learn to map the real input to the forced columns,
        building the sensorimotor forward model.
        """
        cols = self._rng.choice(self.n_columns, self.k_columns, replace=False)

        if encoding is not None:
            # Force high voltage on chosen columns so forced_columns wins
            input_lam = self.input_lamina
            n_per = input_lam.n_per_col
            for col in cols:
                input_lam.voltage[col * n_per : (col + 1) * n_per] = 1.0
            return super().process(encoding, forced_columns=cols)

        # No encoding: just force columns via step directly
        drive = np.zeros(self.input_lamina.n_total)
        n_per = self.input_lamina.n_per_col
        for col in cols:
            drive[col * n_per : (col + 1) * n_per] = 1.0
        return self.step(drive, forced_columns=cols)

    # _learn_ff() inherited from CorticalRegion base class.
    # Base dispatches to _learn_ff_three_factor which handles
    # eligibility trace decay + accumulation.

    def apply_reward(self, reward: float) -> None:
        """Consolidate eligibility traces into weights using reward signal.

        Extends base class with reward baseline subtraction, and
        consolidation of output_weights (L5) and goal_weights (PFC->M1).

        Three-factor rule: dw = (reward - baseline) * eligibility_trace
        Positive reward strengthens recent pathways, negative weakens.
        """
        if not self.learning_enabled:
            return

        # Subtract running baseline (adaptive: good performance -> less change)
        effective_reward = reward
        if self._reward_baseline_decay > 0:
            self._reward_baseline += (1.0 - self._reward_baseline_decay) * (
                reward - self._reward_baseline
            )
            effective_reward = reward - self._reward_baseline

        if abs(effective_reward) < self.REWARD_DEAD_ZONE:
            return

        # Clip motor-specific eligibility traces before consolidation
        if self._eligibility_clip > 0:
            np.clip(
                self._output_eligibility,
                -self._eligibility_clip,
                self._eligibility_clip,
                out=self._output_eligibility,
            )
            if hasattr(self, "_goal_eligibility"):
                np.clip(
                    self._goal_eligibility,
                    -self._eligibility_clip,
                    self._eligibility_clip,
                    out=self._goal_eligibility,
                )

        # Base: clip ff_eligibility + consolidate ff_weights
        super().apply_reward(effective_reward)

        # Consolidate output_weights (L5 readout)
        self.output_weights += effective_reward * self._output_eligibility
        self.output_weights *= self.output_mask
        np.clip(self.output_weights, 0, 1, out=self.output_weights)

        # Consolidate goal_weights (PFC->M1/M2 feedforward)
        # Scaled down (0.3x) for stability -- goal mapping needs to
        # develop slowly to avoid oscillation during echo training.
        if self._goal_weights is not None and hasattr(self, "_goal_eligibility"):
            scale = self.goal_consolidation_scale
            self._goal_weights += scale * effective_reward * self._goal_eligibility
            np.clip(self._goal_weights, 0, 1, out=self._goal_weights)

    def observe_token(self, token_id: int) -> None:
        """Learn L5 output weights: L5 -> token association.

        Three-factor Hebbian on output weights:
        - Record L5->token coincidence in eligibility trace
        - Consolidation happens in apply_reward()
        Also does direct Hebbian strengthening (two-factor baseline)
        so the mapping develops even without reward signal.
        """
        # Map token_id -> L5 output index
        if self._output_vocab is not None:
            matches = np.where(self._output_vocab == token_id)[0]
            if len(matches) == 0:
                return  # Token not in vocabulary
            out_idx = int(matches[0])
        else:
            if token_id < 0 or token_id >= self.n_output_tokens:
                return
            out_idx = token_id

        # Decay output eligibility trace
        self._output_eligibility *= self.eligibility_decay

        # Record coincidence: active L5 neurons -> observed token
        active = self.l5.active.astype(np.float64)
        if not active.any():
            return

        self._output_eligibility[:, out_idx] += active

        # Baseline two-factor Hebbian: small direct weight update
        self.output_weights[:, out_idx] += (
            self.output_lr * 0.1 * active * self.output_mask[:, out_idx]
        )
        np.clip(self.output_weights, 0, 1, out=self.output_weights)

    def get_output(self) -> tuple[int, float]:
        """Return (predicted_token_id, confidence) from output weights.

        Computes token scores from L2/3 activations through learned
        output weights. Same principle as ff_weights but reversed:
        active L2/3 neurons vote for tokens via weighted connections.
        """
        return self.get_population_output()

    def get_population_output(self) -> tuple[int, float]:
        """Return (predicted_token_id, confidence) via output weights.

        L5 activations -> output_weights -> token scores -> argmax.
        Models L5 population coding: the motor command is encoded by
        the pattern of active L5 neurons, decoded through learned
        output weights with structural sparsity.

        Returns (-1, 0.0) if no L5 neurons are active or all scores zero.
        """
        active = self.l5.active.astype(np.float64)
        if not active.any():
            return (-1, 0.0)

        # Token scores: sum of output weights from active L5 neurons
        token_scores = active @ self.output_weights  # (n_output_tokens,)

        if token_scores.max() <= 0:
            return (-1, 0.0)

        best_idx = int(token_scores.argmax())
        # Map L5 output index -> actual token_id
        if self._output_vocab is not None:
            best_token = int(self._output_vocab[best_idx])
        else:
            best_token = best_idx
        # Confidence: normalized score (softmax-like)
        confidence = float(token_scores[best_idx] / max(token_scores.sum(), 1e-10))
        return (best_token, confidence)

    def get_decoded_output(self, decoder) -> tuple[int, float]:
        """Return (predicted_token_id, confidence) using a dendritic decoder.

        L5 readout + BG gating decides IF M1 speaks (output_scores >= threshold).
        The decoder decides WHAT M1 says (replaces column->token frequency map).
        """
        above = self.output_scores >= self.output_threshold
        if not above.any():
            return (-1, 0.0)

        predictions = decoder.decode(self.l23.active, k=1)
        if not predictions:
            return (-1, 0.0)

        confidence = float(self.output_scores[above].max())
        return (predictions[0], confidence)

    def get_output_distribution(self) -> np.ndarray:
        """Return per-column output scores (for visualization)."""
        return self.output_scores.copy()

    def reset_working_memory(self):
        """Reset transient state, preserving learned mappings."""
        super().reset_working_memory()
        self._output_eligibility[:] = 0.0
