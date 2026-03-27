"""Circuit: declarative region wiring + pure neural processing.

Build a circuit by adding regions and connections, then call process()
to run one step of neural computation. Training loops, encoding, and
metrics live in ChatEnv + ChatAgent + train().
"""

from __future__ import annotations

from collections import deque

import numpy as np

from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.circuit_hooks import RunHooks, StepHooks
from step.cortex.circuit_types import (
    Connection,
    ConnectionRole,
    CortexResult,
    Encoder,
    RunMetrics,
    _RegionState,
)
from step.cortex.lamina import Lamina
from step.cortex.modulators import RewardModulator, SurpriseTracker, ThalamicGate
from step.cortex.motor import MotorRegion
from step.cortex.region import CorticalRegion
from step.decoders import DendriticDecoder, InvertedIndexDecoder, SynapticDecoder
from step.probes.diagnostics import CortexDiagnostics
from step.probes.representation import RepresentationTracker
from step.probes.timeline import Timeline

# Re-export types so external code can import from step.cortex.circuit.
__all__ = [
    "Circuit",
    "Connection",
    "ConnectionRole",
    "CortexResult",
    "Encoder",
    "RunHooks",
    "RunMetrics",
    "StepHooks",
]


class Circuit:
    """Declarative region circuit with pure neural processing via process()."""

    def __init__(
        self,
        encoder: Encoder,
        *,
        enable_timeline: bool = False,
        timeline_interval: int = 1,
        diagnostics_interval: int = 100,
        decoder_perm_decay: float = 0.9999,
    ):
        self._encoder = encoder
        self._enable_timeline = enable_timeline
        self._timeline_interval = max(1, timeline_interval)
        self._diagnostics_interval = diagnostics_interval
        self._decoder_perm_decay = decoder_perm_decay
        self._regions: dict[str, _RegionState] = {}
        self._connections: list[Connection] = []
        self._entry_name: str | None = None
        self._finalized: bool = False
        self._topo_cache: list[str] | None = None

        # Persistent turn-taking state (survives across run() calls)
        self._in_eom = False
        self._eom_steps = 0
        # When True, BG gate is forced to 1.0 (open) — for interactive use
        self.force_gate_open = False
        # Tracks total steps across run() calls (BG skips t=0 globally)
        self._total_steps = 0
        # Pluggable reward source (None = use default turn-taking reward)
        self._reward_source = None
        # Pre-allocated BG context buffer (lazily sized)
        self._bg_ctx_buffer: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Finalization (DAG validation)
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Validate the circuit DAG and lock for execution.

        After finalize():
        - No more add_region() or connect() calls
        - Topological order is computed and cached
        - Feedforward dimension mismatches are detected
        - Cycles in feedforward graph raise an error

        run/step/run_echo/etc. auto-finalize if not already done.
        """
        if self._finalized:
            return

        if self._entry_name is None:
            raise ValueError("No entry region. Call add_region(..., entry=True).")

        # Build feedforward adjacency for cycle detection + topo sort
        # Skip self-loops (e.g. M1→M1 modulator-only connections)
        adj: dict[str, list[str]] = {name: [] for name in self._regions}
        for conn in self._connections:
            if conn.role == ConnectionRole.FEEDFORWARD and conn.source != conn.target:
                adj[conn.source].append(conn.target)

        # Kahn's algorithm for topological sort (detects cycles)
        in_degree: dict[str, int] = {name: 0 for name in self._regions}
        for targets in adj.values():
            for t in targets:
                in_degree[t] += 1

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._regions):
            visited = set(order)
            cycle_nodes = [n for n in self._regions if n not in visited]
            raise ValueError(
                f"Feedforward cycle detected involving: {cycle_nodes}. "
                "Cycles are only allowed in apical (feedback) connections."
            )

        # Validate feedforward dimensions: for each target, compute
        # total concatenated input dim and check against ff_weights.
        # Accounts for temporal buffers (buffer_depth multiplies signal).
        for name, s in self._regions.items():
            if name == self._entry_name:
                continue  # Entry gets encoder input, not ff
            ff_dims = []
            for conn in self._connections:
                if (
                    conn.target == name
                    and conn.role == ConnectionRole.FEEDFORWARD
                    and conn.source != conn.target  # skip self-loops
                ):
                    src = self._regions[conn.source].region
                    src_lamina = src.get_lamina(conn.source_lamina)
                    dim = src_lamina.n_total * max(conn.buffer_depth, 1)
                    ff_dims.append(dim)
            if ff_dims:
                total_dim = sum(ff_dims)
                expected = s.region.input_dim
                if total_dim != expected:
                    sources = [
                        f"{c.source}(x{c.buffer_depth})"
                        if c.buffer_depth > 1
                        else c.source
                        for c in self._connections
                        if c.target == name
                        and c.role == ConnectionRole.FEEDFORWARD
                        and c.source != c.target
                    ]
                    raise ValueError(
                        f"Feedforward dimension mismatch for {name}: "
                        f"sources {sources} provide {ff_dims} "
                        f"(total {total_dim}), "
                        f"but {name} expects input_dim={expected}"
                    )

        self._topo_cache = order

        # Pre-compute ff connection lists per target for fast propagation.
        # Single-ff targets get direct pass-through (no concatenation).
        # Multi-ff targets get a pre-allocated buffer.
        # Skip self-loops (modulator-only, no signal flow).
        self._ff_conns: dict[str, list[Connection]] = {}
        self._ff_buffers: dict[str, np.ndarray] = {}
        for name, _s in self._regions.items():
            conns = [
                c
                for c in self._connections
                if c.target == name
                and c.role == ConnectionRole.FEEDFORWARD
                and c.source != c.target
            ]
            if conns:
                self._ff_conns[name] = conns
                if len(conns) > 1:
                    # Pre-allocate concatenation buffer
                    total = sum(
                        self._regions[c.source]
                        .region.get_lamina(c.source_lamina)
                        .n_total
                        * max(c.buffer_depth, 1)
                        for c in conns
                    )
                    self._ff_buffers[name] = np.empty(total, dtype=np.float64)

        # Initialize per-connection traces for temporal credit.
        for conn in self._connections:
            if conn.trace_decay > 0:
                src = self._regions[conn.source].region
                src_lam = src.get_lamina(conn.source_lamina)
                conn._trace = np.zeros(src_lam.n_total, dtype=np.float64)

        self._finalized = True

    # ------------------------------------------------------------------
    # Stage configuration API
    # ------------------------------------------------------------------

    def freeze_region(self, name: str) -> None:
        """Disable all learning in a region (forward pass still runs)."""
        self._regions[name].region.learning_enabled = False

    def unfreeze_region(self, name: str) -> None:
        """Re-enable learning in a region."""
        self._regions[name].region.learning_enabled = True

    def disable_connection(
        self, source: str, target: str, role: ConnectionRole
    ) -> None:
        """Disable a specific connection (signal stops flowing)."""
        for conn in self._connections:
            if conn.source == source and conn.target == target and conn.role == role:
                conn.enabled = False
                return
        raise ValueError(f"No {role.value} connection {source}->{target}")

    def enable_connection(self, source: str, target: str, role: ConnectionRole) -> None:
        """Re-enable a specific connection."""
        for conn in self._connections:
            if conn.source == source and conn.target == target and conn.role == role:
                conn.enabled = True
                return
        raise ValueError(f"No {role.value} connection {source}->{target}")

    def set_reward_source(self, source) -> None:
        """Set a pluggable reward source for BG learning.

        The source should have a step(char, s2_active_columns) method
        that returns a reward float or None. Pass None to revert to
        default turn-taking reward.
        """
        self._reward_source = source

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_region(
        self,
        name: str,
        region: CorticalRegion,
        *,
        entry: bool = False,
        diagnostics: bool = True,
        basal_ganglia: BasalGanglia | None = None,
    ) -> Circuit:
        """Register a region. Exactly one must have entry=True."""
        if self._finalized:
            raise RuntimeError(
                "Circuit is finalized. Cannot add regions after finalize()."
            )
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
                perm_decay=self._decoder_perm_decay,
            )

        if state.motor:
            state.motor_decoder = DendriticDecoder(
                source_dim=region.n_l23_total,
                n_segments=16,
                n_synapses=48,
                perm_decay=self._decoder_perm_decay,
            )

        # Word decoder for non-entry regions (S2, S3, PFC, M2)
        if not entry:
            from step.decoders.word import WordDecoder

            state.word_decoder = WordDecoder(
                region.n_l23_total, seed=hash(name) % (2**31)
            )

        self._regions[name] = state
        return self

    def connect(
        self,
        source: Lamina,
        target: Lamina,
        role: ConnectionRole = ConnectionRole.FEEDFORWARD,
        *,
        surprise_tracker: SurpriseTracker | None = None,
        reward_modulator: RewardModulator | None = None,
        buffer_depth: int = 1,
        burst_gate: bool = False,
        thalamic_gate: ThalamicGate | None = None,
    ) -> Circuit:
        """Wire source lamina -> target lamina.

        Routing is explicit — pass the Lamina objects directly::

            circuit.connect(s1.l23, s2.l4, ConnectionRole.FEEDFORWARD)

        Region names and lamina IDs are derived from the Lamina's
        back-reference to its parent region.
        """
        if self._finalized:
            raise RuntimeError(
                "Circuit is finalized. Cannot add connections after finalize()."
            )
        source_lamina = source.id
        source_name = self._resolve_region_name(source.region)
        target_lamina = target.id
        target_name = self._resolve_region_name(target.region)
        if not isinstance(role, ConnectionRole):
            raise ValueError(f"Unknown connection role: {role!r}")

        conn = Connection(
            source=source_name,
            target=target_name,
            role=role,
            source_lamina=source_lamina,
            target_lamina=target_lamina,
            surprise_tracker=surprise_tracker,
            reward_modulator=reward_modulator,
            buffer_depth=buffer_depth,
            burst_gate=burst_gate,
            thalamic_gate=thalamic_gate,
        )
        if role == ConnectionRole.APICAL:
            tgt_region = self._regions[target_name].region
            tgt_region.init_apical_segments(
                source_dim=source.n_total,
                source_name=source_name,
                target_lamina=target_lamina,
            )

        # Allocate temporal buffer for feedforward connections
        if (
            role == ConnectionRole.FEEDFORWARD
            and buffer_depth > 1
            and source_name != target_name
        ):
            tgt_region = self._regions[target_name].region
            expected_dim = buffer_depth * source.n_total
            if tgt_region.input_dim != expected_dim:
                raise ValueError(
                    f"Target {target_name!r} "
                    f"input_dim={tgt_region.input_dim} "
                    f"but buffer_depth={buffer_depth} * "
                    f"source {source_lamina.value} "
                    f"n_total={source.n_total} = {expected_dim}"
                )
            conn._buffer = np.zeros((buffer_depth, source.n_total))

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

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def region(self, name: str) -> CorticalRegion:
        return self._regions[name].region

    def _resolve_region_name(self, region: CorticalRegion) -> str:
        """Look up the registered name for a region."""
        for name, state in self._regions.items():
            if state.region is region:
                return name
        raise ValueError(f"Region {region!r} not registered in this circuit")

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(
        self, encoding: np.ndarray, *, motor_active: bool | None = None
    ) -> np.ndarray:
        """Process one encoded input through the hierarchy.

        Pure neural processing: takes an encoding vector, propagates
        through regions in topo order, runs inter-region signals, and
        returns the motor output vector (M1 L5 firing rate).

        Encoding/decoding between environment tokens and vectors is the
        caller's responsibility. Boundary and EOM handling use
        reset() and mark_eom() respectively.

        Token-level concerns (observe_token, decoder training, metrics)
        belong at the Agent or Runner level — process() only does
        neural computation.

        Args:
            encoding: Input vector (from encoder or previous circuit output).
            motor_active: Whether motor regions produce output this step.
                If None (default), falls back to self._in_eom or
                self.force_gate_open for backward compatibility.
                Callers should pass this explicitly — the fallback
                will be removed when EOM state moves out of Circuit.

        Returns:
            Motor output vector (M1 L5 firing rate). Callers can decode
            this into an action via a decoder. If no motor region exists,
            returns the L2/3 firing rate of the last region in topo order.
        """
        if self._entry_name is None:
            raise ValueError("No entry region. Call add_region(..., entry=True).")

        entry_name = self._entry_name
        entry_region = self._regions[entry_name].region

        # -- Turn-taking state (deprecated — will move to Environment) --
        if self._in_eom:
            self._eom_steps += 1
            if self._eom_steps > 20:
                self._in_eom = False

        # Resolve motor_active from legacy state if not passed explicitly
        if motor_active is None:
            motor_active = self._in_eom or self.force_gate_open

        # -- Process in topo order (multi-ff supported) --
        topo_order = self._topo_order()
        self._propagate_feedforward(
            topo_order, entry_name, encoding, motor_active=motor_active
        )

        # -- Inter-region signals (after all regions processed) --
        self._propagate_signals()

        # -- Motor: BG gating + output (no token-level learning) --
        self._step_motor_inline(entry_region, motor_active=motor_active)

        self._total_steps += 1

        # -- Return output vector --
        return self._get_output_vector(topo_order)

    def reset(self, *, hooks: StepHooks | None = None) -> None:
        """Reset working memory and transient state (story/dialogue boundary).

        Clears neural activations, connection buffers, modulator state,
        and encoder position tracking. Does NOT reset learned weights —
        this is a "sleep", not amnesia.

        Called by the environment or runner at dialogue/story boundaries.
        """
        if hooks is not None:
            hooks.on_boundary(self)
        for s in self._regions.values():
            s.region.reset_working_memory()
            if s.basal_ganglia is not None:
                s.basal_ganglia.reset()
        for conn in self._connections:
            if conn._trace is not None:
                conn._trace[:] = 0.0
            if conn._buffer is not None:
                conn._buffer[:] = 0.0
                conn._buffer_pos = 0
            if conn.thalamic_gate is not None:
                conn.thalamic_gate.reset()
            if conn.reward_modulator is not None:
                conn.reward_modulator.reset()
        _reset = getattr(self._encoder, "reset", None)
        if _reset is not None:
            _reset()
        self._in_eom = False
        self._eom_steps = 0

    def mark_eom(self) -> None:
        """Signal end-of-message (turn boundary).

        Sets the circuit into EOM mode where motor regions become active.
        Called by the environment when the input stream signals a turn
        boundary (e.g. EOM token in chat).
        """
        self._in_eom = True
        self._eom_steps = 0

    def _step_motor_inline(
        self, entry_region: CorticalRegion, *, motor_active: bool = False
    ) -> None:
        """Motor processing: BG gating and output readout.

        Pure neural computation — no token-level learning, no reward
        computation, no efference copy. Those are Agent/Environment
        responsibilities.
        """
        for _name, s in self._regions.items():
            if s.motor:
                assert isinstance(s.region, MotorRegion)
                motor_region = s.region

                if self._total_steps > 0:
                    # BG gating: always step (learns from both phases)
                    gate = 1.0
                    if s.basal_ganglia is not None:
                        precision = (~entry_region.bursting_columns).astype(np.float64)
                        prec_frac = precision.sum() / max(
                            entry_region.n_columns,
                            1,
                        )
                        ctx = self._build_bg_ctx(precision, prec_frac)
                        gate = s.basal_ganglia.step(ctx)
                        if motor_active:
                            gate = 1.0
                        motor_region.output_scores *= gate

                    if motor_active:
                        pop_id, pop_conf = motor_region.get_population_output()
                        m_id, m_conf = pop_id, pop_conf
                    else:
                        m_id, m_conf = -1, 0.0

                    motor_region.last_output = (m_id, m_conf)
                    motor_region.last_gate = gate
                    motor_region.last_reward = 0.0

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

        # Save encoder alphabet for reconstruction
        if hasattr(self._encoder, "_alphabet"):
            state["encoder_alphabet"] = self._encoder._alphabet
        elif hasattr(self._encoder, "_char_to_idx"):
            state["encoder_alphabet"] = "".join(self._encoder._char_to_idx)

        for name, s in self._regions.items():
            r = s.region
            region_data: dict = {
                "ff_weights": r.ff_weights,
                "l4_to_l23_weights": r.l4_to_l23_weights,
                "l23_to_l5_weights": r.l23_to_l5_weights,
                "l4_lat_seg_indices": r.l4_lat_seg_indices,
                "l4_lat_seg_perm": r.l4_lat_seg_perm,
                "l23_seg_indices": r.l23_seg_indices,
                "l23_seg_perm": r.l23_seg_perm,
                "l5_seg_indices": r.l5_seg_indices,
                "l5_seg_perm": r.l5_seg_perm,
            }
            # Apical per-source segment state
            if r._apical_sources:
                apical_data = {}
                for name, src in r._apical_sources.items():
                    if "seg_indices" in src:
                        apical_data[name] = {
                            "seg_indices": src["seg_indices"].copy(),
                            "seg_perm": src["seg_perm"].copy(),
                        }
                region_data["apical_sources"] = apical_data

            # Save ff_eligibility for any region that uses three-factor learning
            if r._ff_eligibility is not None:
                region_data["ff_eligibility"] = r._ff_eligibility

            if isinstance(r, MotorRegion):
                region_data["output_weights"] = r.output_weights
                region_data["output_mask"] = r.output_mask
                region_data["output_eligibility"] = r._output_eligibility

            if s.basal_ganglia is not None:
                region_data["bg_go_weights"] = s.basal_ganglia.go_weights
                region_data["bg_trace"] = s.basal_ganglia._trace

            if s.dendritic_decoder is not None:
                region_data["decoder_neurons"] = s.dendritic_decoder._neurons

            if s.motor_decoder is not None:
                region_data["motor_decoder_neurons"] = s.motor_decoder._neurons

            if s.word_decoder is not None:
                region_data["word_decoder_state"] = {
                    "neurons": s.word_decoder._decoder._neurons,
                    "word_to_id": s.word_decoder._word_to_id,
                    "id_to_word": s.word_decoder._id_to_word,
                    "next_id": s.word_decoder._next_id,
                }

            state["regions"][name] = region_data

        # Connection modulator state
        for conn in self._connections:
            conn_data: dict = {
                "source": conn.source,
                "target": conn.target,
                "role": conn.role.value,
                "source_lamina": conn.source_lamina.value,
                "target_lamina": conn.target_lamina.value,
            }
            if conn.surprise_tracker is not None:
                conn_data["surprise_burst_ema"] = conn.surprise_tracker._burst_ema
                conn_data["surprise_baseline"] = (
                    conn.surprise_tracker.baseline_burst_rate
                )
            if conn.thalamic_gate is not None:
                conn_data["thalamic_burst_ema"] = conn.thalamic_gate._burst_ema
            state["connections"].append(conn_data)

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path: str) -> None:
        """Restore learned weights and state from a checkpoint file.

        The circuit must already be built with the same architecture
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
            if "l4_to_l23_weights" in region_data:
                r.l4_to_l23_weights[:] = region_data["l4_to_l23_weights"]
            if "l23_to_l5_weights" in region_data:
                r.l23_to_l5_weights[:] = region_data["l23_to_l5_weights"]
            # Old checkpoints may have l23_lateral_weights — skip it.
            # L2/3 lateral now uses segments only.
            # Old checkpoints may have fb_seg — skip it (removed).
            r.l4_lat_seg_indices[:] = region_data["l4_lat_seg_indices"]
            r.l4_lat_seg_perm[:] = region_data["l4_lat_seg_perm"]
            r.l23_seg_indices[:] = region_data["l23_seg_indices"]
            r.l23_seg_perm[:] = region_data["l23_seg_perm"]
            if "l5_seg_indices" in region_data:
                r.l5_seg_indices[:] = region_data["l5_seg_indices"]
                r.l5_seg_perm[:] = region_data["l5_seg_perm"]

            # Load apical state (per-source segments)
            if "apical_sources" in region_data:
                for src_name, saved in region_data["apical_sources"].items():
                    if src_name not in r._apical_sources:
                        continue
                    src = r._apical_sources[src_name]
                    if (
                        isinstance(saved, dict)
                        and "seg_indices" in saved
                        and "seg_indices" in src
                    ):
                        src["seg_indices"][:] = saved["seg_indices"]
                        src["seg_perm"][:] = saved["seg_perm"]

            # Restore ff_eligibility for any region that uses three-factor learning
            if "ff_eligibility" in region_data and r._ff_eligibility is not None:
                r._ff_eligibility[:] = region_data["ff_eligibility"]

            if isinstance(r, MotorRegion):
                if "output_weights" in region_data:
                    r.output_weights[:] = region_data["output_weights"]
                    r.output_mask[:] = region_data["output_mask"]
                if "output_eligibility" in region_data:
                    r._output_eligibility[:] = region_data["output_eligibility"]

            if s.basal_ganglia is not None and "bg_go_weights" in region_data:
                s.basal_ganglia.go_weights[:] = region_data["bg_go_weights"]
                s.basal_ganglia._trace[:] = region_data["bg_trace"]

            if s.dendritic_decoder is not None and "decoder_neurons" in region_data:
                s.dendritic_decoder._neurons = region_data["decoder_neurons"]

            if s.motor_decoder is not None and "motor_decoder_neurons" in region_data:
                s.motor_decoder._neurons = region_data["motor_decoder_neurons"]

            if s.word_decoder is not None and "word_decoder_state" in region_data:
                wd = region_data["word_decoder_state"]
                s.word_decoder._decoder._neurons = wd["neurons"]
                s.word_decoder._word_to_id = wd["word_to_id"]
                s.word_decoder._id_to_word = wd["id_to_word"]
                s.word_decoder._next_id = wd["next_id"]

        # Restore connection modulator state.
        # Handles old checkpoints where role was "surprise" or "reward" —
        # match by (source, target) and modulator presence instead.
        for conn_data in state.get("connections", []):
            # Support both old ("kind") and new ("role") checkpoint keys
            saved_role = conn_data.get("role", conn_data.get("kind", ""))
            for conn in self._connections:
                if (
                    conn.source != conn_data["source"]
                    or conn.target != conn_data["target"]
                ):
                    continue
                # Match: same role, or old "surprise"/"reward" mapped to
                # the connection that now carries the modulator.
                role_match = conn.role.value == saved_role
                surprise_match = (
                    saved_role == "surprise" and conn.surprise_tracker is not None
                )
                reward_match = (
                    saved_role == "reward" and conn.reward_modulator is not None
                )
                if role_match or surprise_match or reward_match:
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
                        conn.thalamic_gate._burst_ema = conn_data["thalamic_burst_ema"]
                    break

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _propagate_feedforward(
        self, topo_order, entry_name, encoding=None, *, motor_active=False
    ):
        """Process regions in topo order with feedforward signals.

        Supports multiple feedforward connections to the same target —
        signals are summed before processing. Biologically, cortical
        regions receive convergent feedforward from multiple sources.
        """
        for name in topo_order:
            s = self._regions[name]
            if name == entry_name:
                if encoding is not None:
                    s.region.process(encoding)
            elif s.motor and not (motor_active or s.region.learning_enabled):
                pass  # Skip idle motor region
            else:
                conns = self._ff_conns.get(name)
                if not conns:
                    continue
                # Filter enabled connections
                active = [c for c in conns if c.enabled]
                if not active:
                    continue
                if len(active) == 1:
                    # Single ff: direct pass-through (no allocation)
                    s.region.process(self._get_ff_signal(active[0]))
                else:
                    # Multi ff: fill pre-allocated buffer
                    buf = self._ff_buffers.get(name)
                    if buf is not None:
                        pos = 0
                        for conn in active:
                            sig = self._get_ff_signal(conn)
                            buf[pos : pos + len(sig)] = sig
                            pos += len(sig)
                        s.region.process(buf[:pos])
                    else:
                        # Fallback (shouldn't happen after finalize)
                        s.region.process(
                            np.concatenate([self._get_ff_signal(c) for c in active])
                        )

    def _propagate_signals(self):
        """Propagate inter-region signals (surprise, apical)."""
        for conn in self._connections:
            if not conn.enabled:
                continue
            src = self._regions[conn.source].region
            tgt = self._regions[conn.target].region

            if conn.surprise_tracker is not None:
                n_active = max(int(src.active_columns.sum()), 1)
                n_bursting = int(src.bursting_columns.sum())
                burst_rate = n_bursting / n_active
                modulator = conn.surprise_tracker.update(burst_rate)
                tgt.surprise_modulator = modulator

                # Note: sensory eligibility traces now consolidate every
                # step in _learn_ff (additive, not gated by surprise).
                # Surprise modulator still scales the base learning rate.

            if conn.role == ConnectionRole.APICAL and tgt.has_apical:
                src_lamina = src.get_lamina(conn.source_lamina)
                # Update connection trace
                if conn._trace is not None:
                    conn._trace *= conn.trace_decay
                    conn._trace += src_lamina.firing_rate
                r_active = max(int(src.active_columns.sum()), 1)
                r_bursting = int(src.bursting_columns.sum())
                confidence = 1.0 - (r_bursting / r_active)
                # Use trace for temporal credit if available
                base = (
                    conn._trace if conn._trace is not None else src_lamina.firing_rate
                )
                signal = base * confidence
                if conn.thalamic_gate is not None:
                    tgt_active = max(int(tgt.active_columns.sum()), 1)
                    tgt_bursting = int(tgt.bursting_columns.sum())
                    tgt_burst_rate = tgt_bursting / tgt_active
                    readiness = conn.thalamic_gate.update(tgt_burst_rate)
                    signal = signal * readiness
                tgt.set_apical_context(signal, source_name=conn.source)

    def _build_bg_ctx(self, precision: np.ndarray, prec_frac: float) -> np.ndarray:
        """Build BG context vector using a pre-allocated buffer."""
        n = precision.shape[0]
        if self._bg_ctx_buffer is None or self._bg_ctx_buffer.shape[0] != n + 1:
            self._bg_ctx_buffer = np.empty(n + 1, dtype=np.float64)
        self._bg_ctx_buffer[:n] = precision
        self._bg_ctx_buffer[n] = prec_frac
        return self._bg_ctx_buffer

    def _get_ff_signal(self, conn: Connection) -> np.ndarray:
        """Build the feedforward signal for a connection.

        Updates the connection trace (temporal credit), applies burst
        gating (if enabled), then writes into the temporal buffer.
        """
        src = self._regions[conn.source].region
        lamina = src.get_lamina(conn.source_lamina)
        signal = lamina.firing_rate.copy()

        # Update connection trace (temporal credit for recent activity)
        if conn._trace is not None:
            conn._trace *= conn.trace_decay
            conn._trace += signal

        # Burst gate: zero precisely-predicted columns
        if conn.burst_gate:
            burst_mask = np.repeat(src.bursting_columns, lamina.n_per_col)
            signal *= burst_mask

        # No buffer: direct pass-through
        if conn.buffer_depth <= 1:
            return signal

        # Write into circular buffer, read oldest-first
        assert conn._buffer is not None
        conn._buffer[conn._buffer_pos] = signal
        conn._buffer_pos = (conn._buffer_pos + 1) % conn.buffer_depth
        return np.roll(conn._buffer, -conn._buffer_pos, axis=0).flatten()

    def _get_output_vector(self, topo_order: list[str]) -> np.ndarray:
        """Return the circuit's output vector after processing.

        If a motor region exists, returns its L5 firing rate (motor
        command vector). Otherwise returns L2/3 firing rate of the
        last region in topo order (highest-level sensory representation).
        """
        for name in reversed(topo_order):
            s = self._regions[name]
            if s.motor:
                return s.region.l5.firing_rate
        # No motor region — return last region's L2/3
        last_name = topo_order[-1]
        return self._regions[last_name].region.l23.firing_rate

    def _topo_order(self) -> list[str]:
        """Return cached topological order. Auto-finalizes if needed."""
        if not self._finalized:
            self.finalize()
        assert self._topo_cache is not None
        return self._topo_cache
