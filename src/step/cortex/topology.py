"""Topology: declarative region wiring that replaces boilerplate run loops.

Build a topology by adding regions and connections, then call run() once.
Supports single-region, two-region hierarchy, and arbitrary DAGs.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import RewardModulator, SurpriseTracker, ThalamicGate
from step.cortex.motor import MotorRegion
from step.cortex.sensory import SensoryRegion
from step.data import EOM_TOKEN, STORY_BOUNDARY
from step.decoders import DendriticDecoder, InvertedIndexDecoder, SynapticDecoder
from step.probes.diagnostics import CortexDiagnostics
from step.probes.representation import RepresentationTracker
from step.probes.timeline import Timeline


class Encoder(Protocol):
    """Minimal encoder interface for the runner."""

    def encode(self, token: str) -> "np.ndarray": ...


@dataclass
class RunMetrics:
    overlaps: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    synaptic_accuracies: list[float] = field(default_factory=list)
    column_accuracies: list[float] = field(default_factory=list)
    dendritic_accuracies: list[float] = field(default_factory=list)
    motor_accuracies: list[float] = field(default_factory=list)
    motor_decoder_accuracies: list[float] = field(default_factory=list)
    motor_population_accuracies: list[float] = field(default_factory=list)
    motor_confidences: list[float] = field(default_factory=list)
    motor_rewards: list[float] = field(default_factory=list)
    # Basal ganglia gate values (when BG is wired)
    bg_gate_values: list[float] = field(default_factory=list)
    # Turn-taking behavioral metrics (Stage 1 RL)
    turn_interruptions: int = 0  # Spoke during input phase
    turn_unresponsive: int = 0   # Silent during EOM phase
    turn_correct_speak: int = 0  # Spoke during EOM phase
    turn_correct_silent: int = 0  # Silent during input phase
    turn_rambles: int = 0        # Spoke past max_speak_steps
    turn_eom_steps: int = 0      # Total steps in EOM phase
    turn_input_steps: int = 0    # Total steps in input phase
    # Bits-per-character (entry region only, uses dendritic decoder)
    bpc: float = 0.0
    bpc_recent: float = 0.0
    # Per-dialogue BPC breakdown (forgetting diagnosis)
    bpc_per_dialogue: list[float] = field(default_factory=list)
    bpc_boundary: list[float] = field(default_factory=list)
    bpc_steady: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    representation: dict = field(default_factory=dict)


@dataclass
class _RegionState:
    """Per-region bookkeeping created by add_region()."""

    region: SensoryRegion
    rep_tracker: RepresentationTracker
    diagnostics: CortexDiagnostics | None
    timeline: Timeline | None
    entry: bool = False
    motor: bool = False
    basal_ganglia: BasalGanglia | None = None
    # Entry region only:
    decode_index: InvertedIndexDecoder | None = None
    syn_decoder: SynapticDecoder | None = None
    dendritic_decoder: DendriticDecoder | None = None
    # Motor region decoder (maps M1 L2/3 → token predictions):
    motor_decoder: DendriticDecoder | None = None


@dataclass
class Connection:
    source: str
    target: str
    kind: str  # "feedforward" | "surprise" | "apical" | "reward"
    surprise_tracker: SurpriseTracker | None = None
    reward_modulator: RewardModulator | None = None
    buffer_depth: int = 1
    burst_gate: bool = False
    thalamic_gate: ThalamicGate | None = None
    _buffer: np.ndarray | None = field(default=None, repr=False)
    _buffer_pos: int = 0


@dataclass
class CortexResult:
    per_region: dict[str, RunMetrics] = field(default_factory=dict)
    surprise_modulators: dict[str, list[float]] = field(default_factory=dict)
    thalamic_readiness: dict[str, list[float]] = field(default_factory=dict)
    reward_modulators: dict[str, list[float]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class Topology:
    """Declarative region topology with a single run() loop."""

    def __init__(
        self,
        encoder: Encoder,
        *,
        enable_timeline: bool = False,
        diagnostics_interval: int = 100,
    ):
        self._encoder = encoder
        self._enable_timeline = enable_timeline
        self._diagnostics_interval = diagnostics_interval
        self._regions: dict[str, _RegionState] = {}
        self._connections: list[Connection] = []
        self._entry_name: str | None = None

        # Persistent turn-taking state (survives across run() calls)
        self._in_eom = False
        self._eom_steps = 0
        # When True, BG gate is forced to 1.0 (open) — for interactive use
        self.force_gate_open = False
        # Tracks total steps across run() calls (BG skips t=0 globally)
        self._total_steps = 0

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_region(
        self,
        name: str,
        region: SensoryRegion,
        *,
        entry: bool = False,
        diagnostics: bool = True,
        basal_ganglia: BasalGanglia | None = None,
    ) -> "Topology":
        """Register a region. Exactly one must have entry=True."""
        if name in self._regions:
            raise ValueError(f"Duplicate region name: {name!r}")
        if entry:
            if self._entry_name is not None:
                raise ValueError(
                    f"Multiple entry regions: {self._entry_name!r} and {name!r}"
                )
            self._entry_name = name

        diag = (
            CortexDiagnostics(snapshot_interval=self._diagnostics_interval)
            if diagnostics
            else None
        )
        timeline = Timeline() if self._enable_timeline else None

        state = _RegionState(
            region=region,
            rep_tracker=RepresentationTracker(region.n_columns, region.n_l4),
            diagnostics=diag,
            timeline=timeline,
            entry=entry,
            motor=isinstance(region, MotorRegion),
            basal_ganglia=basal_ganglia,
        )
        if entry:
            state.decode_index = InvertedIndexDecoder()
            state.syn_decoder = SynapticDecoder()
            state.dendritic_decoder = DendriticDecoder(
                source_dim=region.n_l23_total,
                n_segments=16,
                n_synapses=48,
            )

        if state.motor:
            state.motor_decoder = DendriticDecoder(
                source_dim=region.n_l23_total,
                n_segments=16,
                n_synapses=48,
            )

        self._regions[name] = state
        return self

    def connect(
        self,
        source: str,
        target: str,
        kind: str = "feedforward",
        *,
        surprise_tracker: SurpriseTracker | None = None,
        reward_modulator: RewardModulator | None = None,
        buffer_depth: int = 1,
        burst_gate: bool = False,
        thalamic_gate: ThalamicGate | None = None,
    ) -> "Topology":
        """Wire source -> target."""
        for name in (source, target):
            if name not in self._regions:
                raise ValueError(f"Unknown region: {name!r}")
        if kind not in ("feedforward", "surprise", "apical", "reward"):
            raise ValueError(f"Unknown connection kind: {kind!r}")

        conn = Connection(
            source=source, target=target, kind=kind,
            buffer_depth=buffer_depth, burst_gate=burst_gate,
            thalamic_gate=thalamic_gate,
        )
        if kind == "surprise":
            conn.surprise_tracker = surprise_tracker or SurpriseTracker()
        if kind == "reward":
            conn.reward_modulator = reward_modulator or RewardModulator()
        if kind == "apical":
            src_region = self._regions[source].region
            tgt_region = self._regions[target].region
            if not tgt_region.has_apical:
                tgt_region.init_apical_segments(source_dim=src_region.n_l23_total)

        # Allocate temporal buffer for feedforward connections
        if kind == "feedforward" and buffer_depth > 1:
            src_region = self._regions[source].region
            tgt_region = self._regions[target].region
            expected_dim = buffer_depth * src_region.n_l23_total
            if tgt_region.input_dim != expected_dim:
                raise ValueError(
                    f"Target {target!r} input_dim={tgt_region.input_dim} "
                    f"but buffer_depth={buffer_depth} * "
                    f"source n_l23_total={src_region.n_l23_total} = {expected_dim}"
                )
            conn._buffer = np.zeros((buffer_depth, src_region.n_l23_total))

        self._connections.append(conn)
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def timelines(self) -> dict[str, Timeline]:
        return {
            name: s.timeline
            for name, s in self._regions.items()
            if s.timeline is not None
        }

    @property
    def diagnostics(self) -> dict[str, CortexDiagnostics]:
        return {
            name: s.diagnostics
            for name, s in self._regions.items()
            if s.diagnostics is not None
        }

    def region(self, name: str) -> SensoryRegion:
        return self._regions[name].region

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(
        self,
        tokens: list[tuple[int, str]],
        *,
        log_interval: int = 100,
        rolling_window: int = 100,
        show_predictions: int = 0,
    ) -> CortexResult:
        if self._entry_name is None:
            raise ValueError("No entry region. Call add_region(..., entry=True).")

        topo_order = self._topo_order()
        entry_name = self._entry_name
        entry_state = self._regions[entry_name]
        entry_region = entry_state.region
        k = entry_region.k_columns

        # Per-region metrics accumulators
        metrics: dict[str, RunMetrics] = {
            name: RunMetrics() for name in self._regions
        }
        # BPC probe (entry region only, uses dendritic decoder)
        bpc_probe = None
        if entry_state.dendritic_decoder:
            from step.probes.bpc import BPCProbe
            bpc_probe = BPCProbe()
        # Per-surprise-connection modulator lists, keyed by target name
        surprise_modulators: dict[str, list[float]] = {}
        thalamic_readiness: dict[str, list[float]] = {}
        reward_modulators: dict[str, list[float]] = {}
        for conn in self._connections:
            if conn.kind == "surprise":
                surprise_modulators[conn.target] = []
            if conn.thalamic_gate is not None:
                thalamic_readiness[f"{conn.source}->{conn.target}"] = []
            if conn.kind == "reward":
                reward_modulators[conn.target] = []

        # Turn-taking state for motor RL (Stage 1)
        # Use persistent instance state so EOM carries across run() calls.
        _max_speak_steps = 20  # Anti-rambling: penalize after this many steps

        prediction_log: list[tuple[str, str, str, str]] = []
        start = time.monotonic()

        for t, (token_id, token_str) in enumerate(tokens):
            # -- Story boundary --
            if token_id == STORY_BOUNDARY:
                if bpc_probe is not None:
                    bpc_probe.dialogue_boundary()
                for s in self._regions.values():
                    s.region.reset_working_memory()
                    s.rep_tracker.reset_context()
                    if s.basal_ganglia is not None:
                        s.basal_ganglia.reset()
                for conn in self._connections:
                    if conn._buffer is not None:
                        conn._buffer[:] = 0.0
                        conn._buffer_pos = 0
                    if conn.thalamic_gate is not None:
                        conn.thalamic_gate.reset()
                if hasattr(self._encoder, "reset"):
                    self._encoder.reset()
                self._in_eom = False
                self._eom_steps = 0
                for conn in self._connections:
                    if conn.kind == "reward" and conn.reward_modulator is not None:
                        conn.reward_modulator.reset()
                continue

            # -- EOM token: signal turn boundary for motor RL --
            if token_id == EOM_TOKEN:
                self._in_eom = True
                self._eom_steps = 0
                continue

            # Track turn-taking state
            if self._in_eom:
                self._eom_steps += 1
                # Auto-exit EOM phase after max speaking steps
                if self._eom_steps > _max_speak_steps:
                    self._in_eom = False

            # -- Entry prediction + decode --
            predicted_neurons = entry_region.get_prediction(k)
            predicted_set = frozenset(int(i) for i in predicted_neurons)

            syn_id, syn_str = entry_state.syn_decoder.decode_synaptic(
                predicted_neurons, entry_region
            )
            col_id, col_str = entry_state.syn_decoder.decode_columns(
                predicted_neurons, entry_region.n_l4
            )
            idx_predicted = entry_state.decode_index.decode(predicted_set)

            idx_str = ""
            if (
                idx_predicted >= 0
                and idx_predicted in entry_state.syn_decoder._token_id_set
            ):
                for i, tid in enumerate(entry_state.syn_decoder._token_ids):
                    if tid == idx_predicted:
                        idx_str = entry_state.syn_decoder._token_strs[i]
                        break

            # Dendritic decoder: reads L2/3 binary activations from previous step
            den_predictions = entry_state.dendritic_decoder.decode(
                entry_region.active_l23
            )
            den_id = den_predictions[0] if den_predictions else -1
            den_str = ""
            if den_id >= 0 and den_id in entry_state.syn_decoder._token_id_set:
                for i, tid in enumerate(entry_state.syn_decoder._token_ids):
                    if tid == den_id:
                        den_str = entry_state.syn_decoder._token_strs[i]
                        break

            # BPC: measure prediction quality before processing this token
            if bpc_probe is not None and t > 0:
                bpc_probe.step(
                    token_id, entry_region.active_l23,
                    entry_state.dendritic_decoder,
                )

            # Snapshot L2/3 binary state before processing (for dendritic decoder)
            prev_l23 = entry_region.active_l23.copy()

            # Snapshot motor L2/3 before processing (for motor decoder training)
            prev_motor_l23: dict[str, np.ndarray] = {}
            for _mn, _ms in self._regions.items():
                if _ms.motor and _ms.motor_decoder is not None:
                    prev_motor_l23[_mn] = _ms.region.active_l23.copy()

            # -- Process in topo order --
            for name in topo_order:
                s = self._regions[name]
                if name == entry_name:
                    encoding = self._encoder.encode(token_str)
                    s.region.process(encoding)
                else:
                    # Find feedforward source
                    for conn in self._connections:
                        if conn.target == name and conn.kind == "feedforward":
                            s.region.process(self._get_ff_signal(conn))
                            break

            # -- Inter-region signals (after all regions processed) --
            for conn in self._connections:
                src = self._regions[conn.source].region
                tgt = self._regions[conn.target].region

                if conn.kind == "surprise":
                    n_active = int(src.active_columns.sum())
                    n_bursting = int(src.bursting_columns.sum())
                    burst_rate = n_bursting / max(n_active, 1)
                    modulator = conn.surprise_tracker.update(burst_rate)
                    tgt.surprise_modulator = modulator
                    surprise_modulators[conn.target].append(modulator)

                elif conn.kind == "apical":
                    if tgt.has_apical:
                        r_active = int(src.active_columns.sum())
                        r_bursting = int(src.bursting_columns.sum())
                        confidence = 1.0 - (r_bursting / max(r_active, 1))
                        signal = src.firing_rate_l23 * confidence
                        if conn.thalamic_gate is not None:
                            tgt_active = int(tgt.active_columns.sum())
                            tgt_bursting = int(tgt.bursting_columns.sum())
                            tgt_burst_rate = tgt_bursting / max(tgt_active, 1)
                            readiness = conn.thalamic_gate.update(tgt_burst_rate)
                            signal = signal * readiness
                            key = f"{conn.source}->{conn.target}"
                            thalamic_readiness[key].append(readiness)
                        tgt.set_apical_context(signal)

            # -- Per-region bookkeeping --
            for _name, s in self._regions.items():
                s.rep_tracker.observe(
                    token_id, s.region.active_columns, s.region.active_l4
                )
                if s.diagnostics is not None:
                    s.diagnostics.step(t, s.region)
                if s.timeline is not None:
                    s.timeline.capture(
                        len(s.timeline.frames),
                        s.region,
                        s.region.last_column_drive,
                    )

            # -- Motor metrics + reward --
            for _name, s in self._regions.items():
                if s.motor:
                    motor_region = s.region
                    motor_region.observe_token(token_id)
                    if self._total_steps > 0:
                        # BG gating: step with S1 context, gate M1 output
                        if s.basal_ganglia is not None:
                            # BG context: per-column precision state (inverted
                            # burst). Precise = 1 means column predicted
                            # correctly (EOM/familiar), burst = 0 means novel.
                            # During EOM: dense 1s → strong context signal.
                            # During input: sparse 1s → weak signal.
                            # Models L5/6 → striatum precision projection.
                            precision = (
                                ~entry_region.bursting_columns
                            ).astype(np.float64)
                            prec_frac = precision.sum() / max(
                                entry_region.n_columns, 1,
                            )
                            ctx = np.append(precision, prec_frac)
                            gate = s.basal_ganglia.step(ctx)
                            if self.force_gate_open:
                                gate = 1.0
                            motor_region.output_scores *= gate
                            metrics[_name].bg_gate_values.append(gate)

                        # Compute output from both methods for comparison
                        pop_id, pop_conf = motor_region.get_population_output()
                        if s.motor_decoder is not None:
                            dec_id, _dec_conf = (
                                motor_region.get_decoded_output(
                                    s.motor_decoder,
                                )
                            )
                        else:
                            dec_id = -1

                        # Population vote is primary: biologically grounded
                        # (L5 population coding) and empirically better.
                        # Decoder tracked for diagnostic comparison only.
                        m_id, m_conf = pop_id, pop_conf

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

                        # -- Stage 1 reward: turn-taking --
                        spoke = m_id >= 0
                        reward = self._compute_turn_reward(
                            spoke, self._in_eom, self._eom_steps,
                            _max_speak_steps,
                        )
                        metrics[_name].motor_rewards.append(reward)

                        # Expose last-step state for interactive use
                        motor_region.last_output = (m_id, m_conf)
                        motor_region.last_gate = (
                            gate if s.basal_ganglia is not None
                            else 1.0
                        )
                        motor_region.last_reward = reward

                        # Send gate error signal to BG every step.
                        # Stage 1 is supervised (phase labels known):
                        #   EOM phase: target=1 → signal = 1-gate (push open)
                        #   Input phase: target=0 → signal = -gate (push closed)
                        # Provides gradient from both phases, avoids
                        # sparse-reward collapse. Stage 2/3 will switch to RL.
                        if s.basal_ganglia is not None:
                            gate_target = 1.0 if self._in_eom else 0.0
                            gate_error = gate_target - s.basal_ganglia.gate_value
                            s.basal_ganglia.reward(gate_error)

                        # -- Turn-taking behavioral counters --
                        m = metrics[_name]
                        if self._in_eom:
                            m.turn_eom_steps += 1
                            if spoke:
                                if self._eom_steps > _max_speak_steps:
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

                        # Apply reward through reward connections
                        # (cortical modulation — kept for backward compat)
                        for conn in self._connections:
                            if (
                                conn.kind == "reward"
                                and conn.source == _name
                                and conn.reward_modulator is not None
                            ):
                                mod = conn.reward_modulator.update(reward)
                                tgt = self._regions[conn.target].region
                                tgt.reward_modulator = mod
                                reward_modulators[conn.target].append(
                                    mod
                                )

                        # Efference copy: when M1 output is fed back as
                        # the next input (autoregressive generation), tell
                        # S1 what to expect so it can suppress the predicted
                        # sensory consequence. Only when gate is forced open
                        # (interactive generation) — during training, the
                        # next input comes from the corpus, not M1.
                        if m_id >= 0 and self.force_gate_open:
                            ef_encoding = self._encoder.encode(
                                chr(m_id) if m_id < 128 else "",
                            )
                            entry_region.set_efference_copy(ef_encoding)

                        # Train motor decoder: previous M1 L2/3 → current token
                        if (
                            s.motor_decoder is not None
                            and _name in prev_motor_l23
                        ):
                            s.motor_decoder.observe(
                                token_id, prev_motor_l23[_name],
                            )

            # -- Entry metrics --
            active_set = frozenset(
                int(i) for i in np.nonzero(entry_region.active_l4)[0]
            )

            if t > 0:
                if active_set:
                    overlap = len(predicted_set & active_set) / len(active_set)
                else:
                    overlap = 0.0
                metrics[entry_name].overlaps.append(overlap)

                accuracy = 1.0 if idx_predicted == token_id else 0.0
                metrics[entry_name].accuracies.append(accuracy)

                syn_acc = 1.0 if syn_id == token_id else 0.0
                metrics[entry_name].synaptic_accuracies.append(syn_acc)

                col_acc = 1.0 if col_id == token_id else 0.0
                metrics[entry_name].column_accuracies.append(col_acc)

                den_acc = 1.0 if den_id == token_id else 0.0
                metrics[entry_name].dendritic_accuracies.append(den_acc)

                if show_predictions > 0:
                    prediction_log.append(
                        (token_str, den_str, idx_str, col_str, syn_str)
                    )

            entry_state.decode_index.observe(token_id, active_set)
            entry_state.syn_decoder.observe(
                token_id, token_str, encoding, entry_region.active_columns
            )
            entry_state.dendritic_decoder.observe(token_id, prev_l23)

            self._total_steps += 1

            # -- Logging --
            if (
                t > 0
                and t % log_interval == 0
                and metrics[entry_name].overlaps
            ):
                self._log_step(
                    t, start, entry_name, metrics, surprise_modulators,
                    thalamic_readiness, reward_modulators, rolling_window,
                    show_predictions, prediction_log, bpc_probe,
                )

        elapsed = time.monotonic() - start

        # -- Finalize per-region representation summaries --
        for name, s in self._regions.items():
            m = metrics[name]
            m.elapsed_seconds = elapsed
            rep_summ = s.rep_tracker.summary(s.region.ff_weights)
            sel = s.rep_tracker.column_selectivity()
            rep_summ["column_selectivity_per_col"] = sel["per_column"]
            m.representation = rep_summ

        # Store BPC in entry metrics
        if bpc_probe is not None:
            # Flush last dialogue
            bpc_probe.dialogue_boundary()
            entry_m = metrics[entry_name]
            entry_m.bpc = bpc_probe.bpc
            entry_m.bpc_recent = bpc_probe.recent_bpc
            entry_m.bpc_per_dialogue = bpc_probe.dialogue_bpcs
            entry_m.bpc_boundary = bpc_probe.boundary_bpcs
            entry_m.bpc_steady = bpc_probe.steady_bpcs

        # Print representation reports
        if len(self._regions) == 1:
            entry_state.rep_tracker.print_report(entry_region.ff_weights)
        else:
            for name, s in self._regions.items():
                print(f"\n--- {name} ---")
                s.rep_tracker.print_report(s.region.ff_weights)

        return CortexResult(
            per_region=metrics,
            surprise_modulators=surprise_modulators,
            thalamic_readiness=thalamic_readiness,
            reward_modulators=reward_modulators,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save learned weights and state to a file.

        Captures all learned state needed to resume without retraining:
        region weights, segment permanences, motor mappings, BG weights,
        decoder neurons, and modulator EMA state.
        """
        import pickle

        state: dict = {"regions": {}, "connections": []}

        for name, s in self._regions.items():
            r = s.region
            region_data: dict = {
                "ff_weights": r.ff_weights,
                "l23_lateral_weights": r.l23_lateral_weights,
                "fb_seg_indices": r.fb_seg_indices,
                "fb_seg_perm": r.fb_seg_perm,
                "lat_seg_indices": r.lat_seg_indices,
                "lat_seg_perm": r.lat_seg_perm,
                "l23_seg_indices": r.l23_seg_indices,
                "l23_seg_perm": r.l23_seg_perm,
            }
            if r.apical_seg_indices is not None:
                region_data["apical_seg_indices"] = r.apical_seg_indices
                region_data["apical_seg_perm"] = r.apical_seg_perm

            if isinstance(r, MotorRegion):
                region_data["_col_token_counts"] = r._col_token_counts
                region_data["_col_token_map"] = r._col_token_map

            if s.basal_ganglia is not None:
                region_data["bg_go_weights"] = s.basal_ganglia.go_weights
                region_data["bg_trace"] = s.basal_ganglia._trace

            if s.dendritic_decoder is not None:
                region_data["decoder_neurons"] = s.dendritic_decoder._neurons

            if s.motor_decoder is not None:
                region_data["motor_decoder_neurons"] = s.motor_decoder._neurons

            state["regions"][name] = region_data

        # Connection modulator state
        for conn in self._connections:
            conn_data: dict = {
                "source": conn.source,
                "target": conn.target,
                "kind": conn.kind,
            }
            if conn.surprise_tracker is not None:
                conn_data["surprise_burst_ema"] = (
                    conn.surprise_tracker._burst_ema
                )
                conn_data["surprise_baseline"] = (
                    conn.surprise_tracker.baseline_burst_rate
                )
            if conn.thalamic_gate is not None:
                conn_data["thalamic_burst_ema"] = (
                    conn.thalamic_gate._burst_ema
                )
            state["connections"].append(conn_data)

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path: str) -> None:
        """Restore learned weights and state from a checkpoint file.

        The topology must already be built with the same architecture
        (same regions, connections, dimensions) before loading.
        """
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        for name, region_data in state["regions"].items():
            if name not in self._regions:
                continue
            s = self._regions[name]
            r = s.region

            r.ff_weights[:] = region_data["ff_weights"]
            r.l23_lateral_weights[:] = region_data["l23_lateral_weights"]
            r.fb_seg_indices[:] = region_data["fb_seg_indices"]
            r.fb_seg_perm[:] = region_data["fb_seg_perm"]
            r.lat_seg_indices[:] = region_data["lat_seg_indices"]
            r.lat_seg_perm[:] = region_data["lat_seg_perm"]
            r.l23_seg_indices[:] = region_data["l23_seg_indices"]
            r.l23_seg_perm[:] = region_data["l23_seg_perm"]

            if (
                "apical_seg_indices" in region_data
                and r.apical_seg_indices is not None
            ):
                r.apical_seg_indices[:] = region_data["apical_seg_indices"]
                r.apical_seg_perm[:] = region_data["apical_seg_perm"]

            if (
                isinstance(r, MotorRegion)
                and "_col_token_counts" in region_data
            ):
                r._col_token_counts = region_data["_col_token_counts"]
                r._col_token_map[:] = region_data["_col_token_map"]

            if (
                s.basal_ganglia is not None
                and "bg_go_weights" in region_data
            ):
                s.basal_ganglia.go_weights[:] = region_data[
                    "bg_go_weights"
                ]
                s.basal_ganglia._trace[:] = region_data["bg_trace"]

            if (
                s.dendritic_decoder is not None
                and "decoder_neurons" in region_data
            ):
                s.dendritic_decoder._neurons = region_data[
                    "decoder_neurons"
                ]

            if (
                s.motor_decoder is not None
                and "motor_decoder_neurons" in region_data
            ):
                s.motor_decoder._neurons = region_data[
                    "motor_decoder_neurons"
                ]

        # Restore connection modulator state
        for conn_data in state.get("connections", []):
            for conn in self._connections:
                if (
                    conn.source == conn_data["source"]
                    and conn.target == conn_data["target"]
                    and conn.kind == conn_data["kind"]
                ):
                    if (
                        conn.surprise_tracker is not None
                        and "surprise_burst_ema" in conn_data
                    ):
                        conn.surprise_tracker._burst_ema = conn_data[
                            "surprise_burst_ema"
                        ]
                        conn.surprise_tracker.baseline_burst_rate = conn_data[
                            "surprise_baseline"
                        ]
                    if (
                        conn.thalamic_gate is not None
                        and "thalamic_burst_ema" in conn_data
                    ):
                        conn.thalamic_gate._burst_ema = conn_data[
                            "thalamic_burst_ema"
                        ]
                    break

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_turn_reward(
        spoke: bool,
        in_eom: bool,
        eom_steps: int,
        max_speak_steps: int,
    ) -> float:
        """Stage 1 turn-taking reward.

        Rewards:
          +0.5  M1 speaks during EOM phase (correct turn-taking)
          -0.5  M1 speaks during input (should be listening)
          +0.2  M1 silent during input (correct listening)
          -0.3  M1 silent during EOM (should be speaking)
          -1.0  M1 speaks past max steps (rambling penalty)

        Returns reward in [-1.0, +0.5] range.
        """
        if in_eom:
            if eom_steps > max_speak_steps and spoke:
                return -1.0  # Rambling
            return 0.5 if spoke else -0.3
        else:
            return -0.5 if spoke else 0.2

    def _get_ff_signal(self, conn: Connection) -> np.ndarray:
        """Build the feedforward signal for a connection.

        Applies burst gating (if enabled) then writes into the temporal
        buffer (if depth > 1), returning the concatenated oldest-first
        window.
        """
        src = self._regions[conn.source].region
        signal = src.firing_rate_l23.copy()

        # Burst gate: zero precisely-predicted columns
        if conn.burst_gate:
            burst_mask = np.repeat(src.bursting_columns, src.n_l23)
            signal *= burst_mask

        # No buffer: direct pass-through
        if conn.buffer_depth <= 1:
            return signal

        # Write into circular buffer, read oldest-first
        conn._buffer[conn._buffer_pos] = signal
        conn._buffer_pos = (conn._buffer_pos + 1) % conn.buffer_depth
        return np.roll(conn._buffer, -conn._buffer_pos, axis=0).flatten()

    def _topo_order(self) -> list[str]:
        """BFS from entry following feedforward edges."""
        adj: dict[str, list[str]] = {name: [] for name in self._regions}
        for conn in self._connections:
            if conn.kind == "feedforward":
                adj[conn.source].append(conn.target)

        order: list[str] = []
        visited: set[str] = set()
        queue: deque[str] = deque()
        queue.append(self._entry_name)
        visited.add(self._entry_name)

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Add any regions not reachable via feedforward (e.g. disconnected)
        for name in self._regions:
            if name not in visited:
                order.append(name)

        return order

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
        prediction_log: list[tuple[str, ...]],
        bpc_probe: object = None,
    ):
        entry_metrics = metrics[entry_name]
        entry_diag = self._regions[entry_name].diagnostics

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

        label = entry_name if len(self._regions) > 1 else "cortex"

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
        for _name, s in self._regions.items():
            if s.motor:
                m = metrics[_name]
                if m.motor_accuracies:
                    tail_m = m.motor_accuracies[-rolling_window:]
                    roll_m = sum(tail_m) / len(tail_m)
                    # Silence rate: steps with confidence 0 / total steps
                    tail_c = m.motor_confidences[-rolling_window:]
                    silence = (
                        sum(1 for c in tail_c if c == 0.0)
                        / max(len(tail_c), 1)
                    )
                    motor_str += f" M1={roll_m:.4f} sil={silence:.0%}"
                    # Compare decoder vs population accuracy
                    if m.motor_decoder_accuracies:
                        tail_dec = m.motor_decoder_accuracies[
                            -rolling_window:
                        ]
                        roll_dec = sum(tail_dec) / len(tail_dec)
                        motor_str += f" dec={roll_dec:.4f}"
                    if m.motor_population_accuracies:
                        tail_pop = m.motor_population_accuracies[
                            -rolling_window:
                        ]
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
                        intr = (
                            m.turn_interruptions / inp_t
                            if inp_t > 0 else 0
                        )
                        unre = (
                            m.turn_unresponsive / eom_t
                            if eom_t > 0 else 0
                        )
                        motor_str += (
                            f" int={intr:.0%}"
                            f" unr={unre:.0%}"
                        )
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
                    "*" if p == actual else " "
                    for p in (den_p, idx_p, col_p, syn_p)
                ]
                print(
                    f"    {fmt(actual)} "
                    f"|{marks[0]}{fmt(den_p)} "
                    f"|{marks[1]}{fmt(idx_p)} "
                    f"|{marks[2]}{fmt(col_p)} "
                    f"|{marks[3]}{fmt(syn_p)}"
                )
            prediction_log.clear()
