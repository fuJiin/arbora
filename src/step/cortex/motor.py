"""Motor cortical region: maps sensory context to output predictions.

Subclasses SensoryRegion (encoding_width=0, full connectivity) and adds
an L5-like output layer with confidence gating. Columns self-organize
to represent output tokens via Hebbian learning — same cortical algorithm
as sensory regions, different wiring.

The column→token mapping is learned: each column tracks which token_id
most frequently activates it, forming a self-organizing motor map.
"""

import numpy as np

from step.cortex.region import CorticalRegion


class MotorRegion(CorticalRegion):
    """Cortical region with L5 output layer for token production.

    Receives S1's L2/3 firing rate as input (full connectivity, no
    encoding-width structure). Adds:
    - L5 output layer: learned weights mapping L2/3 → token scores
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
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            n_columns=n_columns,
            seed=seed,
            **kwargs,
        )
        self.output_threshold = output_threshold

        # Vocabulary: maps L5 output index → token_id.
        # If provided, L5 only produces valid tokens.
        # If None, output index IS the token_id (raw ASCII).
        if output_vocab is not None:
            self._output_vocab = np.array(output_vocab, dtype=np.int64)
            n_output_tokens = len(output_vocab)
        else:
            self._output_vocab = None

        # Babbling mode: mix random sparse drive with normal feedforward.
        # 0.0 = normal (no noise), 1.0 = pure random babbling.
        # Models brainstem pattern generators in early infant vocalization.
        self.babbling_noise: float = 0.0

        # Three-factor learning: eligibility trace on ff_weights.
        self._ff_eligibility = np.zeros_like(self.ff_weights)
        self._eligibility_decay = 0.95  # ~20-step window

        # L5 output state (updated each step in process())
        self.output_scores = np.zeros(n_columns)

        # Goal drive: additive feedforward signal from PFC.
        # Added to ff_weights drive before k-WTA competition.
        # This is the "direct dial" — PFC→M1 feedforward shortcut
        # for simple responses. Will be replaced by PFC→M2→M1 later.
        self._goal_drive: np.ndarray | None = None
        self._goal_weights: np.ndarray | None = None

        # -- L5 output layer: L2/3 → token scores --
        # Models cortical L5 pyramidal neurons that project to motor
        # periphery. Learned weights with structural sparsity, same
        # architecture as ff_weights. Three-factor Hebbian learning:
        # pre=L2/3 activity, post=token, modulated by reward.
        self.n_output_tokens = n_output_tokens
        self.output_lr = output_lr
        n_l23 = self.n_l23_total

        # L5 weights: (n_l23_total, n_output_tokens)
        self.output_weights = self._rng.uniform(
            0,
            0.01,
            size=(n_l23, n_output_tokens),
        )
        # Structural sparsity: each L2/3 neuron connects to ~50% of tokens
        self.output_mask = (
            self._rng.random((n_l23, n_output_tokens)) < output_sparsity
        ).astype(np.float64)
        self.output_weights *= self.output_mask

        # L5 eligibility trace (three-factor learning)
        self._output_eligibility = np.zeros((n_l23, n_output_tokens))

        # Last step output (set by topology run loop after BG gating)
        self.last_output: tuple[int, float] = (-1, 0.0)
        self.last_gate: float = 0.5
        self.last_reward: float = 0.0

    def init_goal_drive(self, source_dim: int) -> None:
        """Initialize PFC→M1 feedforward goal weights.

        Called once when PFC is connected. Creates learned weights that
        map PFC's L2/3 firing rate to additive M1 neuron drive.
        """
        self._goal_weights = self._rng.uniform(
            0,
            0.01,
            size=(source_dim, self.n_l4_total),
        )
        # Structural sparsity: ~50% connectivity
        goal_mask = self._rng.random((source_dim, self.n_l4_total)) < 0.5
        self._goal_weights *= goal_mask
        self._goal_eligibility = np.zeros_like(self._goal_weights)
        self._goal_drive = None

    def set_goal_drive(self, pfc_firing_rate: np.ndarray) -> None:
        """Set PFC goal signal for next process() call."""
        self._goal_drive = pfc_firing_rate

    def process(self, encoding: np.ndarray) -> np.ndarray:
        """Feedforward + goal drive + L5 output scores.

        Combines S1 feedforward (sensory context) with PFC goal drive
        (what to produce). Both are additive on neuron voltage before
        k-WTA competition selects winning columns.
        """
        if self.babbling_noise >= 1.0:
            active = self._babble_direct(encoding)
        elif self.babbling_noise > 0.0:
            if self._rng.random() < self.babbling_noise:
                active = self._babble_direct(encoding)
            else:
                active = self._process_with_goal(encoding)
        else:
            active = self._process_with_goal(encoding)

        # L5 readout: per-column mean L2/3 firing rate
        rates = self.firing_rate_l23.reshape(self.n_columns, self.n_l23)
        self.output_scores = rates.mean(axis=1)

        return active

    def _process_with_goal(self, encoding: np.ndarray) -> np.ndarray:
        """Normal feedforward processing with additive PFC goal drive."""
        flat = encoding.flatten().astype(np.float64)
        neuron_drive = flat @ self.ff_weights

        # Add PFC goal drive (if set)
        if self._goal_drive is not None and self._goal_weights is not None:
            goal_signal = self._goal_drive
            goal_drive = goal_signal @ self._goal_weights
            neuron_drive += goal_drive

            # Record goal eligibility (three-factor: PFC activity x M1 winners)
            if hasattr(self, "_goal_eligibility"):
                self._goal_eligibility *= self._eligibility_decay
                # Will be populated after step() determines winners
                self._pending_goal_signal = goal_signal

            self._goal_drive = None  # Consumed

        self.last_column_drive = neuron_drive.reshape(self.n_columns, self.n_l4).max(
            axis=1
        )
        active = self.step(neuron_drive)

        # Record goal eligibility for winner neurons
        if hasattr(self, "_goal_eligibility") and hasattr(self, "_pending_goal_signal"):
            winner_cols = np.nonzero(self.active_columns)[0]
            if len(winner_cols) > 0:
                winner_neurons = []
                for col in winner_cols:
                    winner_neurons.extend(range(col * self.n_l4, (col + 1) * self.n_l4))
                # Slow learning rate for goal weights (stability over speed)
                goal_lr = self.learning_rate * 0.1
                self._goal_eligibility[:, winner_neurons] += (
                    goal_lr * self._pending_goal_signal[:, np.newaxis]
                )
            del self._pending_goal_signal

        if self.learning_enabled:
            self._learn_ff(flat)
        return active

    def _babble_direct(self, encoding: np.ndarray | None = None) -> np.ndarray:
        """Force random column activations while training ff_weights.

        Randomly selects k columns (ensuring diverse exploration), then
        runs normal L2/3 activation and learning. Crucially, also trains
        ff_weights via _learn_ff so the sensorimotor mapping develops —
        M1 learns "when S1 pattern is X, I should activate columns Y."

        Args:
            encoding: The real feedforward input (S1 L2/3). If provided,
                      ff_weights are trained to map this input to the
                      forced columns. Builds the forward model.
        """
        # Pick random k columns
        cols = self._rng.choice(self.n_columns, self.k_columns, replace=False)

        # Run the standard step pipeline but inject our columns
        # 1. Decay voltages
        self.voltage_l4 *= self.voltage_decay
        self.voltage_l23 *= self.voltage_decay

        # 2. Compute predictions (for segment learning)
        self._compute_predictions()

        # 3. Save prediction context
        self._pred_context_l23[:] = self.active_l23
        self._pred_context_l4[:] = self.active_l4

        # 4-6. Force our chosen columns active
        for col in cols:
            start = col * self.n_l4
            end = start + self.n_l4
            self.voltage_l4[start:end] = 1.0

        scores_l4 = self.voltage_l4 + self.excitability_l4
        self._activate_l4_burst(cols, scores_l4)

        # 7. Activate L2/3
        self._activate_l23(cols)

        # 8. Learn — segments AND ff_weights
        if self.learning_enabled:
            self._learn()
            # Train ff_weights: learn to map S1 input → forced columns.
            # This builds the sensorimotor forward model.
            if encoding is not None:
                flat = encoding.flatten().astype(np.float64)
                self._learn_ff(flat)

        # 9-13. Standard housekeeping
        self._update_traces()
        self._update_excitability()
        self.voltage_l4[self.active_l4] = 0.0
        self.voltage_l23[self.active_l23] = 0.0
        np.clip(self.voltage_l4, 0.0, 1.0, out=self.voltage_l4)
        np.clip(self.voltage_l23, 0.0, 1.0, out=self.voltage_l23)
        self.firing_rate_l23 *= self.voltage_decay
        self.firing_rate_l23[self.active_l23] += 1.0 - self.voltage_decay

        return np.nonzero(self.active_l4)[0]

    def _learn_ff(self, flat_input: np.ndarray):
        """Three-factor Hebbian learning on ff_weights.

        Instead of applying weight changes immediately (two-factor Hebbian),
        records changes in an eligibility trace. Changes are consolidated
        when apply_reward() is called with the reward signal.

        This lets reward determine whether recent motor activations were
        good (consolidate) or bad (reverse).
        """
        # Decay eligibility trace
        self._ff_eligibility *= self._eligibility_decay

        # Compute what the standard Hebbian update WOULD be
        active_cols = np.nonzero(self.active_columns)[0]
        if len(active_cols) == 0:
            return

        # Find winner neurons
        voltage_by_col = self.voltage_l4.reshape(self.n_columns, self.n_l4)
        active_by_col = self.active_l4.reshape(self.n_columns, self.n_l4)
        is_burst = self.bursting_columns[active_cols]

        winner_indices = np.empty(len(active_cols), dtype=np.intp)
        if is_burst.any():
            winner_indices[is_burst] = active_cols[
                is_burst
            ] * self.n_l4 + voltage_by_col[active_cols[is_burst]].argmax(axis=1)
        precise = ~is_burst
        if precise.any():
            winner_indices[precise] = active_cols[precise] * self.n_l4 + active_by_col[
                active_cols[precise]
            ].argmax(axis=1)

        # Record Hebbian coincidence in eligibility trace (not weights)
        # LTP direction: input * post-activity
        self._ff_eligibility[:, winner_indices] += (
            self.learning_rate * flat_input[:, np.newaxis]
        )

    def apply_reward(self, reward: float) -> None:
        """Consolidate eligibility traces into weights using reward signal.

        Three-factor rule: dw = reward * eligibility_trace
        Positive reward → strengthen recent pathways
        Negative reward → weaken them

        Applies to both:
        - ff_weights (input → column mapping)
        - output_weights (L2/3 → token mapping, L5)
        """
        if not self.learning_enabled:
            return
        if abs(reward) < 1e-6:
            return

        # Consolidate ff_weights (input mapping)
        self.ff_weights += reward * self._ff_eligibility
        self.ff_weights *= self.ff_mask
        np.clip(self.ff_weights, 0, 1, out=self.ff_weights)

        # Consolidate output_weights (L5 readout)
        self.output_weights += reward * self._output_eligibility
        self.output_weights *= self.output_mask
        np.clip(self.output_weights, 0, 1, out=self.output_weights)

        # Consolidate goal_weights (PFC→M1/M2 feedforward)
        # Scaled down (0.3x) for stability — goal mapping needs to
        # develop slowly to avoid oscillation during echo training.
        if self._goal_weights is not None and hasattr(self, "_goal_eligibility"):
            self._goal_weights += 0.3 * reward * self._goal_eligibility
            np.clip(self._goal_weights, 0, 1, out=self._goal_weights)

    def observe_token(self, token_id: int) -> None:
        """Learn L5 output weights: L2/3 → token association.

        Three-factor Hebbian on output weights:
        - Record L2/3→token coincidence in eligibility trace
        - Consolidation happens in apply_reward()
        Also does direct Hebbian strengthening (two-factor baseline)
        so the mapping develops even without reward signal.
        """
        # Map token_id → L5 output index
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
        self._output_eligibility *= self._eligibility_decay

        # Record coincidence: active L2/3 neurons → observed token
        active = self.active_l23.astype(np.float64)
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

        L2/3 activations → output_weights → token scores → argmax.
        Models L5 population coding: the motor command is encoded by
        the pattern of active L2/3 neurons, decoded through learned
        output weights with structural sparsity.

        Returns (-1, 0.0) if no L2/3 neurons are active or all scores zero.
        """
        active = self.active_l23.astype(np.float64)
        if not active.any():
            return (-1, 0.0)

        # Token scores: sum of output weights from active L2/3 neurons
        token_scores = active @ self.output_weights  # (n_output_tokens,)

        if token_scores.max() <= 0:
            return (-1, 0.0)

        best_idx = int(token_scores.argmax())
        # Map L5 output index → actual token_id
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
        The decoder decides WHAT M1 says (replaces column→token frequency map).
        """
        above = self.output_scores >= self.output_threshold
        if not above.any():
            return (-1, 0.0)

        predictions = decoder.decode(self.active_l23, k=1)
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
        self.output_scores[:] = 0.0
        self._ff_eligibility[:] = 0.0
        self._output_eligibility[:] = 0.0
