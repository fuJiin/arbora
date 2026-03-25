"""Step hooks for the topology system.

Defines the StepHooks protocol and RunHooks — the concrete implementation
that extracts all metrics, BPC, logging, and decoder logic from run().
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Protocol

import numpy as np

from step.cortex.motor import MotorRegion
from step.cortex.topology_types import (
    ConnectionRole,
    CortexResult,
    RunMetrics,
)

if TYPE_CHECKING:
    from step.cortex.topology import Topology
    from step.probes.bpc import BPCProbe
    from step.probes.centroid_bpc import CentroidBPCProbe


class StepHooks(Protocol):
    """Protocol for observing step() execution.

    All methods are optional no-ops. Implement any subset to observe
    or instrument the training loop.
    """

    def on_before_step(
        self, topology: Topology, t: int, token_id: int, token_str: str
    ) -> None: ...

    def on_after_step(
        self, topology: Topology, t: int, token_id: int, token_str: str
    ) -> None: ...

    def on_boundary(self, topology: Topology) -> None: ...


class RunHooks:
    """Concrete StepHooks that implements all metrics/BPC/logging from run().

    This is a "friend class" of Topology — it reads internal state directly.
    """

    # Anti-rambling: penalize after this many steps in EOM phase
    MAX_SPEAK_STEPS = 20
    # Motor learning also considers this for M1 active check
    MAX_EOM_STEPS_FOR_LEARNING = 10

    def __init__(
        self,
        topology: Topology,
        *,
        log_interval: int = 100,
        rolling_window: int = 100,
        show_predictions: int = 0,
        metric_interval: int = 0,
    ):
        self._topology = topology

        self._log_interval = log_interval
        self._rolling_window = rolling_window
        self._show_predictions = show_predictions
        # metric_interval controls how often expensive decode/prediction
        # metrics are computed. Default (0) = every log_interval steps.
        if metric_interval <= 0:
            metric_interval = max(1, log_interval)
        self._metric_interval = metric_interval

        entry_name = topology._entry_name
        assert entry_name is not None
        self._entry_name = entry_name
        entry_state = topology._regions[entry_name]
        entry_region = entry_state.region
        self._k = entry_region.k_columns

        # Per-region metrics accumulators
        self._metrics: dict[str, RunMetrics] = {
            name: RunMetrics() for name in topology._regions
        }

        # BPC probes (entry region only)
        self._bpc_probe: BPCProbe | None = None
        if entry_state.dendritic_decoder:
            from step.probes.bpc import BPCProbe

            self._bpc_probe = BPCProbe()
        from step.probes.centroid_bpc import CentroidBPCProbe

        self._centroid_probe: CentroidBPCProbe = CentroidBPCProbe(
            source_dim=entry_region.n_l23_total
        )

        # Per-surprise-connection modulator lists, keyed by target name
        self._surprise_modulators: dict[str, list[float]] = {}
        self._thalamic_readiness: dict[str, list[float]] = {}
        self._reward_modulators: dict[str, list[float]] = {}
        for conn in topology._connections:
            if conn.surprise_tracker is not None:
                self._surprise_modulators[conn.target] = []
            if conn.thalamic_gate is not None:
                self._thalamic_readiness[f"{conn.source}->{conn.target}"] = []
            if conn.reward_modulator is not None:
                self._reward_modulators[conn.target] = []

        self._prediction_log: list[tuple[str, str, str, str, str]] = []
        self._start = time.monotonic()

        # Per-step state set by on_before_step, used by on_after_step
        self._prev_l23: np.ndarray | None = None
        self._m1_active: bool = False
        self._prev_motor_l23: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # StepHooks interface
    # ------------------------------------------------------------------

    def on_boundary(self, topology: Topology) -> None:
        """BPC dialogue boundary, rep_tracker reset, reward resets.

        Called BEFORE step()'s own boundary handling (region resets etc.).
        """
        if self._bpc_probe is not None:
            self._bpc_probe.dialogue_boundary()
        self._centroid_probe.dialogue_boundary()

        for s in topology._regions.values():
            s.rep_tracker.reset_context()

        for conn in topology._connections:
            if conn.reward_modulator is not None:
                conn.reward_modulator.reset()

        if topology._reward_source is not None:
            _rs_reset = getattr(topology._reward_source, "reset", None)
            if _rs_reset is not None:
                _rs_reset()

    def on_before_step(
        self, topology: Topology, t: int, token_id: int, token_str: str
    ) -> None:
        """Snapshot L2/3 state before processing."""
        entry_region = topology._regions[self._entry_name].region

        # Snapshot L2/3 binary state before processing (for dendritic decoder)
        self._prev_l23 = entry_region.active_l23.copy()

        # Motor regions process when: EOM phase, gate forced open, or
        # learning enabled (listening phase -- M1 observes to build
        # internal representations before babbling).
        m1_active = topology._in_eom or topology.force_gate_open
        for _mn, _ms in topology._regions.items():
            if _ms.motor and _ms.region.learning_enabled:
                m1_active = True
                break
        self._m1_active = m1_active

        # Snapshot motor L2/3 before processing (for motor decoder training)
        self._prev_motor_l23 = {}
        if m1_active:
            for _mn, _ms in topology._regions.items():
                if _ms.motor and _ms.motor_decoder is not None:
                    self._prev_motor_l23[_mn] = _ms.region.active_l23.copy()

    def on_after_step(
        self, topology: Topology, t: int, token_id: int, token_str: str
    ) -> None:
        """All per-step metrics, decoder training, logging."""
        entry_name = self._entry_name
        entry_state = topology._regions[entry_name]
        entry_region = entry_state.region
        metrics = self._metrics
        prev_l23 = self._prev_l23
        assert prev_l23 is not None

        # -- Inter-region signal metric recording --
        # The actual propagation happened in step(). We read values from
        # the connection objects to record time series.
        for conn in topology._connections:
            if not conn.enabled:
                continue
            if conn.surprise_tracker is not None:
                # Read the last modulator value from the tracker
                modulator = conn.surprise_tracker.modulator
                self._surprise_modulators[conn.target].append(modulator)

            if conn.role == ConnectionRole.APICAL and conn.thalamic_gate is not None:
                tgt = topology._regions[conn.target].region
                if tgt.has_apical:
                    readiness = conn.thalamic_gate.readiness
                    key = f"{conn.source}->{conn.target}"
                    self._thalamic_readiness[key].append(readiness)

        # -- Per-region bookkeeping (sampled to reduce overhead) --
        is_metric_step = (t % self._metric_interval == 0) or (t < 100)
        for _name, s in topology._regions.items():
            if is_metric_step:
                s.rep_tracker.observe(
                    token_id, s.region.active_columns, s.region.active_l4
                )
            if s.diagnostics is not None and is_metric_step:
                s.diagnostics.step(t, s.region)
            if s.timeline is not None and t % topology._timeline_interval == 0:
                s.timeline.capture(
                    len(s.timeline.frames),
                    s.region,
                    s.region.last_column_drive,
                )

        # -- Motor metrics + reward --
        self._process_motor_metrics(topology, t, token_id, token_str, metrics)

        # -- Entry metrics (expensive decodes sampled at metric intervals) --
        if is_metric_step and t > 0:
            self._process_entry_metrics(topology, t, token_id, token_str, metrics)

        # BPC: measure prediction quality (sampled at metric intervals)
        if self._bpc_probe is not None and is_metric_step and t > 0:
            assert entry_state.dendritic_decoder is not None
            self._bpc_probe.step(
                token_id,
                entry_region.active_l23,
                entry_state.dendritic_decoder,
            )
        if is_metric_step and t > 0:
            self._centroid_probe.step(token_id, prev_l23)
        self._centroid_probe.observe(token_id, prev_l23)

        # -- Decoder training (every step -- cheap, drives learning) --
        self._train_decoders(topology, t, token_id, token_str, prev_l23)

        # -- Logging --
        if (
            t > 0
            and t % self._log_interval == 0
            and metrics[entry_name].dendritic_accuracies
        ):
            self._log_step(
                t,
                self._start,
                entry_name,
                metrics,
                self._surprise_modulators,
                self._thalamic_readiness,
                self._reward_modulators,
                self._rolling_window,
                self._show_predictions,
                self._prediction_log,
                self._bpc_probe,
                self._centroid_probe,
            )

    # ------------------------------------------------------------------
    # Finalize (post-loop)
    # ------------------------------------------------------------------

    def finalize(self) -> CortexResult:
        """Post-loop work: representation summaries, BPC flush, result."""
        topology = self._topology
        entry_name = self._entry_name
        metrics = self._metrics

        elapsed = time.monotonic() - self._start

        # -- Finalize per-region representation summaries --
        for name, s in topology._regions.items():
            m = metrics[name]
            m.elapsed_seconds = elapsed
            rep_summ = s.rep_tracker.summary(s.region.ff_weights)
            sel = s.rep_tracker.column_selectivity()
            rep_summ["column_selectivity_per_col"] = sel["per_column"]
            m.representation = rep_summ

        # Store BPC in entry metrics
        if self._bpc_probe is not None:
            # Flush last dialogue
            self._bpc_probe.dialogue_boundary()
            entry_m = metrics[entry_name]
            entry_m.bpc = self._bpc_probe.bpc
            entry_m.bpc_recent = self._bpc_probe.recent_bpc
            entry_m.bpc_per_dialogue = self._bpc_probe.dialogue_bpcs
            entry_m.bpc_boundary = self._bpc_probe.boundary_bpcs
            entry_m.bpc_steady = self._bpc_probe.steady_bpcs

        # Store centroid BPC
        self._centroid_probe.dialogue_boundary()
        entry_m = metrics[entry_name]
        entry_m.centroid_bpc = self._centroid_probe.bpc
        entry_m.centroid_bpc_recent = self._centroid_probe.recent_bpc

        # Print representation reports
        if len(topology._regions) == 1:
            entry_state = topology._regions[entry_name]
            entry_state.rep_tracker.print_report(entry_state.region.ff_weights)
        else:
            for name, s in topology._regions.items():
                print(f"\n--- {name} ---")
                s.rep_tracker.print_report(s.region.ff_weights)

        return CortexResult(
            per_region=metrics,
            surprise_modulators=self._surprise_modulators,
            thalamic_readiness=self._thalamic_readiness,
            reward_modulators=self._reward_modulators,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_motor_metrics(
        self,
        topology: Topology,
        t: int,
        token_id: int,
        token_str: str,
        metrics: dict[str, RunMetrics],
    ) -> None:
        """Motor metrics + reward (extracted from run() lines 762-893)."""
        entry_name = self._entry_name
        entry_region = topology._regions[entry_name].region
        m1_active = self._m1_active
        prev_motor_l23 = self._prev_motor_l23
        _max_speak_steps = self.MAX_SPEAK_STEPS

        for _name, s in topology._regions.items():
            if s.motor:
                assert isinstance(s.region, MotorRegion)
                motor_region = s.region
                # observe_token only during active phase (M1 processed)
                if m1_active:
                    motor_region.observe_token(token_id)
                if topology._total_steps > 0:
                    # BG gating: always step (learns from both phases)
                    gate = 1.0
                    if s.basal_ganglia is not None:
                        precision = (~entry_region.bursting_columns).astype(np.float64)
                        prec_frac = precision.sum() / max(
                            entry_region.n_columns,
                            1,
                        )
                        ctx = topology._build_bg_ctx(precision, prec_frac)
                        gate = s.basal_ganglia.step(ctx)
                        if topology.force_gate_open:
                            gate = 1.0
                        motor_region.output_scores *= gate
                        metrics[_name].bg_gate_values.append(gate)

                    if m1_active:
                        # M1 processed this step -- compute output + metrics
                        pop_id, pop_conf = motor_region.get_population_output()
                        if s.motor_decoder is not None:
                            dec_id, _dec_conf = motor_region.get_decoded_output(
                                s.motor_decoder,
                            )
                        else:
                            dec_id = -1
                        m_id, m_conf = pop_id, pop_conf
                    else:
                        # M1 idle during input -- silent output
                        m_id, m_conf = -1, 0.0
                        pop_id, dec_id = -1, -1

                    metrics[_name].motor_confidences.append(m_conf)
                    if m_id >= 0:
                        metrics[_name].motor_accuracies.append(
                            1.0 if m_id == token_id else 0.0
                        )
                    # Track both methods independently
                    if pop_id >= 0:
                        metrics[_name].motor_population_accuracies.append(
                            1.0 if pop_id == token_id else 0.0,
                        )
                    if dec_id >= 0:
                        metrics[_name].motor_decoder_accuracies.append(
                            1.0 if dec_id == token_id else 0.0,
                        )

                    # -- Motor reward --
                    spoke = m_id >= 0
                    if topology._reward_source is not None:
                        m_char = chr(m_id) if spoke and 32 <= m_id < 127 else None
                        reward = topology._compute_pluggable_reward(
                            m_char,
                            entry_region,
                        )
                    else:
                        reward = topology._compute_turn_reward(
                            spoke,
                            topology._in_eom,
                            topology._eom_steps,
                            _max_speak_steps,
                        )
                    metrics[_name].motor_rewards.append(reward)

                    # Expose last-step state for interactive use
                    motor_region.last_output = (m_id, m_conf)
                    motor_region.last_gate = gate
                    motor_region.last_reward = reward

                    # BG reward: send computed reward to update gate weights
                    if s.basal_ganglia is not None:
                        if topology._reward_source is not None:
                            # Pluggable reward: send directly to BG
                            s.basal_ganglia.reward(reward)
                        else:
                            # Default: turn-taking gate error
                            gate_target = 1.0 if topology._in_eom else 0.0
                            gate_error = gate_target - s.basal_ganglia.gate_value
                            s.basal_ganglia.reward(gate_error)

                    # -- Turn-taking behavioral counters --
                    m = metrics[_name]
                    if topology._in_eom:
                        m.turn_eom_steps += 1
                        if spoke:
                            if topology._eom_steps > _max_speak_steps:
                                m.turn_rambles += 1
                            else:
                                m.turn_correct_speak += 1
                        else:
                            m.turn_unresponsive += 1
                    else:
                        m.turn_input_steps += 1
                        if spoke:
                            m.turn_interruptions += 1
                        else:
                            m.turn_correct_silent += 1

                    # Apply reward through connections with reward modulators
                    for conn in topology._connections:
                        if conn.source == _name and conn.reward_modulator is not None:
                            mod = conn.reward_modulator.update(reward)
                            tgt = topology._regions[conn.target].region
                            tgt.reward_modulator = mod
                            self._reward_modulators[conn.target].append(mod)

                    # Efference copy: only during generation (gate forced open)
                    if m_id >= 0 and topology.force_gate_open:
                        ef_encoding = topology._encoder.encode(
                            chr(m_id) if m_id < 128 else "",
                        )
                        entry_region.set_efference_copy(ef_encoding)

                    # Train motor decoder: previous M1 L2/3 -> current token
                    if s.motor_decoder is not None and _name in prev_motor_l23:
                        s.motor_decoder.observe(
                            token_id,
                            prev_motor_l23[_name],
                        )

    def _process_entry_metrics(
        self,
        topology: Topology,
        t: int,
        token_id: int,
        token_str: str,
        metrics: dict[str, RunMetrics],
    ) -> None:
        """Entry-region metrics: predictions, decoder accuracies."""
        entry_name = self._entry_name
        entry_state = topology._regions[entry_name]
        entry_region = entry_state.region
        k = self._k

        predicted_neurons = entry_region.get_prediction(k)
        active_l4_indices = np.nonzero(entry_region.active_l4)[0]

        # Overlap
        if len(active_l4_indices) > 0 and len(predicted_neurons) > 0:
            n_hit = np.isin(active_l4_indices, predicted_neurons).sum()
            overlap = n_hit / len(active_l4_indices)
        else:
            overlap = 0.0
        metrics[entry_name].overlaps.append(overlap)

        # Decoder accuracies
        assert entry_state.syn_decoder is not None
        assert entry_state.decode_index is not None
        assert entry_state.dendritic_decoder is not None
        syn_id, syn_str = entry_state.syn_decoder.decode_synaptic(
            predicted_neurons, entry_region
        )
        col_id, col_str = entry_state.syn_decoder.decode_columns(
            predicted_neurons, entry_region.n_l4
        )
        predicted_set = frozenset(int(i) for i in predicted_neurons)
        idx_predicted = entry_state.decode_index.decode(predicted_set)
        den_predictions = entry_state.dendritic_decoder.decode(entry_region.active_l23)
        den_id = den_predictions[0] if den_predictions else -1

        metrics[entry_name].accuracies.append(1.0 if idx_predicted == token_id else 0.0)
        metrics[entry_name].synaptic_accuracies.append(
            1.0 if syn_id == token_id else 0.0
        )
        metrics[entry_name].column_accuracies.append(1.0 if col_id == token_id else 0.0)
        metrics[entry_name].dendritic_accuracies.append(
            1.0 if den_id == token_id else 0.0
        )

        if self._show_predictions > 0:
            idx_str = ""
            if (
                idx_predicted >= 0
                and idx_predicted in entry_state.syn_decoder._token_id_set
            ):
                for i, tid in enumerate(entry_state.syn_decoder._token_ids):
                    if tid == idx_predicted:
                        idx_str = entry_state.syn_decoder._token_strs[i]
                        break
            den_str = ""
            if den_id >= 0 and den_id in entry_state.syn_decoder._token_id_set:
                for i, tid in enumerate(entry_state.syn_decoder._token_ids):
                    if tid == den_id:
                        den_str = entry_state.syn_decoder._token_strs[i]
                        break
            self._prediction_log.append((token_str, den_str, idx_str, col_str, syn_str))

    def _train_decoders(
        self,
        topology: Topology,
        t: int,
        token_id: int,
        token_str: str,
        prev_l23: np.ndarray,
    ) -> None:
        """Decoder training (every step -- cheap, drives learning)."""
        entry_name = self._entry_name
        entry_state = topology._regions[entry_name]
        entry_region = entry_state.region
        encoding = topology._encoder.encode(token_str)

        assert entry_state.decode_index is not None
        assert entry_state.syn_decoder is not None
        assert entry_state.dendritic_decoder is not None
        if token_id not in entry_state.decode_index._token_id_to_idx:
            active_set = frozenset(
                int(i) for i in np.nonzero(entry_region.active_l4)[0]
            )
            entry_state.decode_index.observe(token_id, active_set)
        entry_state.syn_decoder.observe(
            token_id, token_str, encoding, entry_region.active_columns
        )
        entry_state.dendritic_decoder.observe(token_id, prev_l23)

        # Train word decoders on all non-entry regions
        for _wd_name, _wd_state in topology._regions.items():
            if _wd_state.word_decoder is not None:
                _wd_state.word_decoder.step(token_str, _wd_state.region.firing_rate_l23)

    def _log_step(
        self,
        t: int,
        start: float,
        entry_name: str,
        metrics: dict[str, RunMetrics],
        surprise_modulators: dict[str, list[float]],
        thalamic_readiness: dict[str, list[float]],
        reward_modulators: dict[str, list[float]],
        rolling_window: int,
        show_predictions: int,
        prediction_log: list[tuple[str, str, str, str, str]],
        bpc_probe: BPCProbe | None = None,
        centroid_probe: CentroidBPCProbe | None = None,
    ) -> None:
        """Periodic logging (moved from Topology._log_step)."""
        topology = self._topology
        entry_metrics = metrics[entry_name]
        entry_diag = topology._regions[entry_name].diagnostics

        tail_den = entry_metrics.dendritic_accuracies[-rolling_window:]
        roll_den = sum(tail_den) / len(tail_den) if tail_den else 0.0
        tail_syn = entry_metrics.synaptic_accuracies[-rolling_window:]
        roll_syn = sum(tail_syn) / len(tail_syn)
        tail_o = entry_metrics.overlaps[-rolling_window:]
        roll_o = sum(tail_o) / len(tail_o)
        elapsed = time.monotonic() - start

        burst_pct = 0.0
        if entry_diag is not None:
            bc = entry_diag._burst_count
            pc = entry_diag._precise_count
            total = bc + pc
            burst_pct = bc / total if total > 0 else 0.0

        label = entry_name if len(topology._regions) > 1 else "cortex"

        # Surprise modulator info
        mod_str = ""
        if surprise_modulators:
            multi = len(surprise_modulators) > 1
            for tgt, mods in surprise_modulators.items():
                if mods:
                    tail_mod = mods[-rolling_window:]
                    avg_mod = sum(tail_mod) / len(tail_mod)
                    tag = f"mod({tgt})" if multi else "mod"
                    mod_str += f" {tag}={avg_mod:.2f}"

        # Thalamic gate readiness
        gate_str = ""
        if thalamic_readiness:
            multi = len(thalamic_readiness) > 1
            for key, vals in thalamic_readiness.items():
                if vals:
                    tail_gate = vals[-rolling_window:]
                    avg_gate = sum(tail_gate) / len(tail_gate)
                    tag = f"gate({key})" if multi else "gate"
                    gate_str += f" {tag}={avg_gate:.2f}"

        # Reward modulator info
        reward_str = ""
        if reward_modulators:
            multi = len(reward_modulators) > 1
            for tgt, rews in reward_modulators.items():
                if rews:
                    tail_rew = rews[-rolling_window:]
                    avg_rew = sum(tail_rew) / len(tail_rew)
                    tag = f"rew({tgt})" if multi else "rew"
                    reward_str += f" {tag}={avg_rew:.2f}"

        # Motor accuracy
        motor_str = ""
        for _name, s in topology._regions.items():
            if s.motor:
                m = metrics[_name]
                if m.motor_accuracies:
                    tail_m = m.motor_accuracies[-rolling_window:]
                    roll_m = sum(tail_m) / len(tail_m)
                    # Silence rate: steps with confidence 0 / total steps
                    tail_c = m.motor_confidences[-rolling_window:]
                    silence = sum(1 for c in tail_c if c == 0.0) / max(len(tail_c), 1)
                    motor_str += f" M1={roll_m:.4f} sil={silence:.0%}"
                    # Compare decoder vs population accuracy
                    if m.motor_decoder_accuracies:
                        tail_dec = m.motor_decoder_accuracies[-rolling_window:]
                        roll_dec = sum(tail_dec) / len(tail_dec)
                        motor_str += f" dec={roll_dec:.4f}"
                    if m.motor_population_accuracies:
                        tail_pop = m.motor_population_accuracies[-rolling_window:]
                        roll_pop = sum(tail_pop) / len(tail_pop)
                        motor_str += f" pop={roll_pop:.4f}"
                    # Average reward
                    if m.motor_rewards:
                        tail_r = m.motor_rewards[-rolling_window:]
                        avg_r = sum(tail_r) / len(tail_r)
                        motor_str += f" r={avg_r:+.3f}"
                    # Turn-taking rates
                    if m.turn_eom_steps > 0 or m.turn_input_steps > 0:
                        eom_t = m.turn_eom_steps
                        inp_t = m.turn_input_steps
                        intr = m.turn_interruptions / inp_t if inp_t > 0 else 0
                        unre = m.turn_unresponsive / eom_t if eom_t > 0 else 0
                        motor_str += f" int={intr:.0%} unr={unre:.0%}"
                        if m.turn_rambles > 0:
                            motor_str += f" ram={m.turn_rambles}"
                    # BG gate value
                    if m.bg_gate_values:
                        tail_g = m.bg_gate_values[-rolling_window:]
                        avg_g = sum(tail_g) / len(tail_g)
                        motor_str += f" bg={avg_g:.2f}"

        # BPC info
        bpc_str = ""
        if bpc_probe is not None and bpc_probe.bpc < float("inf"):
            bpc_str = f" bpc={bpc_probe.recent_bpc:.2f}"
            # Show boundary vs steady-state BPC (last 5 dialogues)
            bdry = bpc_probe.boundary_bpcs[-5:]
            stdy = bpc_probe.steady_bpcs[-5:]
            if bdry and stdy:
                avg_b = sum(bdry) / len(bdry)
                avg_s = sum(stdy) / len(stdy)
                bpc_str += f" bdry={avg_b:.2f} stdy={avg_s:.2f}"
        if (
            centroid_probe is not None
            and centroid_probe.n_tokens > 1
            and centroid_probe.bpc < float("inf")
        ):
            bpc_str += f" cbpc={centroid_probe.recent_bpc:.2f}"
        # Decoder BPC: approximate from dendritic decoder accuracy
        if roll_den > 0.001:
            dbpc = -math.log2(max(roll_den, 1e-10))
            bpc_str += f" dbpc={dbpc:.2f}"

        print(
            f"  [{label}] t={t:,} "
            f"den={roll_den:.4f} "
            f"syn={roll_syn:.4f} "
            f"overlap={roll_o:.4f} "
            f"burst={burst_pct:.1%}"
            f"{bpc_str}{mod_str}{gate_str}{reward_str}{motor_str} "
            f"({elapsed:.1f}s)"
        )

        if show_predictions > 0 and prediction_log:
            samples = prediction_log[-show_predictions:]
            hdr = (
                f"{'actual':>12s} | {'den':>12s} | {'idx':>12s} "
                f"| {'col':>12s} | {'syn':>12s}"
            )
            sep = f"{'-' * 12}-+-" * 4 + f"{'-' * 12}"
            print(f"    {hdr}")
            print(f"    {sep}")
            for actual, den_p, idx_p, col_p, syn_p in samples:
                fmt = lambda s: repr(s)[:12].ljust(12)  # noqa: E731
                marks = [
                    "*" if p == actual else " " for p in (den_p, idx_p, col_p, syn_p)
                ]
                print(
                    f"    {fmt(actual)} "
                    f"|{marks[0]}{fmt(den_p)} "
                    f"|{marks[1]}{fmt(idx_p)} "
                    f"|{marks[2]}{fmt(col_p)} "
                    f"|{marks[3]}{fmt(syn_p)}"
                )
            prediction_log.clear()
