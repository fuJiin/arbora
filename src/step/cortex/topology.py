"""Topology: declarative region wiring that replaces boilerplate run loops.

Build a topology by adding regions and connections, then call run() once.
Supports single-region, two-region hierarchy, and arbitrary DAGs.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.lamina import LaminaID
from step.cortex.modulators import RewardModulator, SurpriseTracker, ThalamicGate
from step.cortex.motor import MotorRegion
from step.cortex.region import CorticalRegion
from step.cortex.topology_types import (
    Connection,
    ConnectionRole,
    CortexResult,
    Encoder,
    RunMetrics,
    _RegionState,
)
from step.data import EOM_TOKEN, STORY_BOUNDARY
from step.decoders import DendriticDecoder, InvertedIndexDecoder, SynapticDecoder
from step.probes.diagnostics import CortexDiagnostics
from step.probes.representation import RepresentationTracker
from step.probes.timeline import Timeline

if TYPE_CHECKING:
    from step.probes.bpc import BPCProbe
    from step.probes.centroid_bpc import CentroidBPCProbe

# Re-export types for backward compatibility.
# External code imports these from step.cortex.topology.
__all__ = [
    "Connection",
    "ConnectionRole",
    "CortexResult",
    "Encoder",
    "RunMetrics",
    "Topology",
]


class Topology:
    """Declarative region topology with a single run() loop."""

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
        """Validate the topology DAG and lock for execution.

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
    ) -> Topology:
        """Register a region. Exactly one must have entry=True."""
        if self._finalized:
            raise RuntimeError(
                "Topology is finalized. Cannot add regions after finalize()."
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
        source: str,
        target: str,
        role: ConnectionRole = ConnectionRole.FEEDFORWARD,
        *,
        source_lamina: LaminaID = LaminaID.L23,
        target_lamina: LaminaID = LaminaID.L4,
        surprise_tracker: SurpriseTracker | None = None,
        reward_modulator: RewardModulator | None = None,
        buffer_depth: int = 1,
        burst_gate: bool = False,
        thalamic_gate: ThalamicGate | None = None,
    ) -> Topology:
        """Wire source -> target.

        Args:
            source_lamina: Which layer's output to read from the source
                region (default L2/3 — corticocortical projection).
            target_lamina: Which layer receives the signal in the target
                region (default L4 for feedforward input).
        """
        if self._finalized:
            raise RuntimeError(
                "Topology is finalized. Cannot add connections after finalize()."
            )
        for name in (source, target):
            if name not in self._regions:
                raise ValueError(f"Unknown region: {name!r}")
        if not isinstance(role, ConnectionRole):
            raise ValueError(f"Unknown connection role: {role!r}")

        conn = Connection(
            source=source,
            target=target,
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
            src_region = self._regions[source].region
            tgt_region = self._regions[target].region
            src_lamina = src_region.get_lamina(source_lamina)
            tgt_region.init_apical_segments(
                source_dim=src_lamina.n_total,
                source_name=source,
            )

        # Allocate temporal buffer for feedforward connections (skip self-loops)
        if role == ConnectionRole.FEEDFORWARD and buffer_depth > 1 and source != target:
            src_region = self._regions[source].region
            tgt_region = self._regions[target].region
            src_lamina = src_region.get_lamina(source_lamina)
            expected_dim = buffer_depth * src_lamina.n_total
            if tgt_region.input_dim != expected_dim:
                raise ValueError(
                    f"Target {target!r} input_dim={tgt_region.input_dim} "
                    f"but buffer_depth={buffer_depth} * "
                    f"source {source_lamina.value} n_total="
                    f"{src_lamina.n_total} = {expected_dim}"
                )
            conn._buffer = np.zeros((buffer_depth, src_lamina.n_total))

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

    def region(self, name: str) -> CorticalRegion:
        return self._regions[name].region

    # ------------------------------------------------------------------
    # Single-token step (lightweight, no metrics overhead)
    # ------------------------------------------------------------------

    def step(self, token_id: int, token_str: str) -> None:
        """Process one token through the hierarchy.

        Lightweight alternative to run() for interactive use and probing.
        Handles EOM/boundary tokens, feedforward processing, inter-region
        signals, motor output, and BG gating — but skips metrics
        accumulation, logging, BPC, and diagnostics.
        """
        if self._entry_name is None:
            raise ValueError("No entry region. Call add_region(..., entry=True).")

        entry_name = self._entry_name
        entry_state = self._regions[entry_name]
        entry_region = entry_state.region

        # -- Story boundary --
        if token_id == STORY_BOUNDARY:
            for s in self._regions.values():
                s.region.reset_working_memory()
                if s.basal_ganglia is not None:
                    s.basal_ganglia.reset()
            for conn in self._connections:
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
            return

        # -- EOM token --
        if token_id == EOM_TOKEN:
            self._in_eom = True
            self._eom_steps = 0
            return

        # -- Turn-taking state --
        if self._in_eom:
            self._eom_steps += 1
            if self._eom_steps > 20:
                self._in_eom = False

        # -- Process in topo order --
        # Ensure finalize() has been called (multi-ff needs _ff_conns)
        if not getattr(self, "_ff_conns", None):
            self.finalize()
        # Motor regions skip process() during input phase (not EOM, gate
        # not forced open). BG/observe still run — only the expensive
        # cortical computation is skipped.
        m1_active = self._in_eom or self.force_gate_open
        encoding = self._encoder.encode(token_str)
        topo_order = self._topo_order()

        # Use _propagate_feedforward for proper multi-ff concatenation
        # (PFC and M2 receive multiple ff sources)
        for name in topo_order:
            s = self._regions[name]
            if s.motor and not m1_active:
                # Skip M1 process during input phase
                pass
            elif name == entry_name:
                s.region.process(encoding)
            else:
                conns = self._ff_conns.get(name)
                if not conns:
                    continue
                active = [c for c in conns if c.enabled]
                if not active:
                    continue
                if len(active) == 1:
                    s.region.process(self._get_ff_signal(active[0]))
                else:
                    buf = self._ff_buffers.get(name)
                    if buf is not None:
                        pos = 0
                        for conn in active:
                            sig = self._get_ff_signal(conn)
                            buf[pos : pos + len(sig)] = sig
                            pos += len(sig)
                        s.region.process(buf[:pos])

        # -- Inter-region signals --
        for conn in self._connections:
            src = self._regions[conn.source].region
            tgt = self._regions[conn.target].region

            if conn.surprise_tracker is not None:
                n_active = int(src.active_columns.sum())
                n_bursting = int(src.bursting_columns.sum())
                burst_rate = n_bursting / max(n_active, 1)
                modulator = conn.surprise_tracker.update(burst_rate)
                tgt.surprise_modulator = modulator

            if conn.role == ConnectionRole.APICAL and tgt.has_apical:
                src_lamina = src.get_lamina(conn.source_lamina)
                r_active = int(src.active_columns.sum())
                r_bursting = int(src.bursting_columns.sum())
                confidence = 1.0 - (r_bursting / max(r_active, 1))
                signal = src_lamina.firing_rate * confidence
                if conn.thalamic_gate is not None:
                    tgt_active = int(tgt.active_columns.sum())
                    tgt_bursting = int(tgt.bursting_columns.sum())
                    tgt_burst_rate = tgt_bursting / max(tgt_active, 1)
                    readiness = conn.thalamic_gate.update(tgt_burst_rate)
                    signal = signal * readiness
                tgt.set_apical_context(signal, source_name=conn.source)

        # -- Motor processing --
        for _name, s in self._regions.items():
            if s.motor:
                assert isinstance(s.region, MotorRegion)
                motor_region = s.region
                if m1_active:
                    motor_region.observe_token(token_id)

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
                        if self.force_gate_open:
                            gate = 1.0
                        motor_region.output_scores *= gate

                    if m1_active:
                        pop_id, pop_conf = motor_region.get_population_output()
                        m_id, m_conf = pop_id, pop_conf
                    else:
                        m_id, m_conf = -1, 0.0

                    motor_region.last_output = (m_id, m_conf)
                    motor_region.last_gate = gate
                    motor_region.last_reward = 0.0

                    # BG reward: always send (learns from both phases)
                    if s.basal_ganglia is not None:
                        gate_target = 1.0 if self._in_eom else 0.0
                        gate_error = gate_target - s.basal_ganglia.gate_value
                        s.basal_ganglia.reward(gate_error)
                        spoke = m_id >= 0
                        reward = self._compute_turn_reward(
                            spoke,
                            self._in_eom,
                            self._eom_steps,
                            20,
                        )
                        motor_region.last_reward = reward
                        for conn in self._connections:
                            if (
                                conn.source == _name
                                and conn.reward_modulator is not None
                            ):
                                mod = conn.reward_modulator.update(reward)
                                self._regions[conn.target].region.reward_modulator = mod

                    # Efference copy (generation mode only)
                    if m_id >= 0 and self.force_gate_open:
                        ef_encoding = self._encoder.encode(
                            chr(m_id) if m_id < 128 else "",
                        )
                        entry_region.set_efference_copy(ef_encoding)

        self._total_steps += 1

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
        metric_interval: int = 0,
    ) -> CortexResult:
        if self._entry_name is None:
            raise ValueError("No entry region. Call add_region(..., entry=True).")

        # metric_interval controls how often expensive decode/prediction
        # metrics are computed. Default (0) = every log_interval steps.
        # Set to 1 for full resolution (slower), or N for every Nth step.
        if metric_interval <= 0:
            metric_interval = max(1, log_interval)

        topo_order = self._topo_order()
        entry_name = self._entry_name
        entry_state = self._regions[entry_name]
        entry_region = entry_state.region
        k = entry_region.k_columns

        # Per-region metrics accumulators
        metrics: dict[str, RunMetrics] = {name: RunMetrics() for name in self._regions}
        # BPC probes (entry region only)
        bpc_probe = None
        if entry_state.dendritic_decoder:
            from step.probes.bpc import BPCProbe

            bpc_probe = BPCProbe()
        from step.probes.centroid_bpc import CentroidBPCProbe

        centroid_probe = CentroidBPCProbe(source_dim=entry_region.n_l23_total)
        # Per-surprise-connection modulator lists, keyed by target name
        surprise_modulators: dict[str, list[float]] = {}
        thalamic_readiness: dict[str, list[float]] = {}
        reward_modulators: dict[str, list[float]] = {}
        for conn in self._connections:
            if conn.surprise_tracker is not None:
                surprise_modulators[conn.target] = []
            if conn.thalamic_gate is not None:
                thalamic_readiness[f"{conn.source}->{conn.target}"] = []
            if conn.reward_modulator is not None:
                reward_modulators[conn.target] = []

        # Turn-taking state for motor RL (Stage 1)
        # Use persistent instance state so EOM carries across run() calls.
        _max_speak_steps = 20  # Anti-rambling: penalize after this many steps

        prediction_log: list[tuple[str, str, str, str, str]] = []
        start = time.monotonic()

        for t, (token_id, token_str) in enumerate(tokens):
            # -- Story boundary --
            if token_id == STORY_BOUNDARY:
                if bpc_probe is not None:
                    bpc_probe.dialogue_boundary()
                centroid_probe.dialogue_boundary()
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
                _reset = getattr(self._encoder, "reset", None)
                if _reset is not None:
                    _reset()
                self._in_eom = False
                self._eom_steps = 0
                for conn in self._connections:
                    if conn.reward_modulator is not None:
                        conn.reward_modulator.reset()
                if self._reward_source is not None:
                    _rs_reset = getattr(self._reward_source, "reset", None)
                    if _rs_reset is not None:
                        _rs_reset()
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

            # Snapshot L2/3 binary state before processing (for dendritic decoder)
            prev_l23 = entry_region.active_l23.copy()

            # Motor regions process when: EOM phase, gate forced open, or
            # learning enabled (listening phase — M1 observes to build
            # internal representations before babbling).
            m1_active = self._in_eom or self.force_gate_open
            for _mn, _ms in self._regions.items():
                if _ms.motor and _ms.region.learning_enabled:
                    m1_active = True
                    break

            # Snapshot motor L2/3 before processing (for motor decoder training)
            prev_motor_l23: dict[str, np.ndarray] = {}
            if m1_active:
                for _mn, _ms in self._regions.items():
                    if _ms.motor and _ms.motor_decoder is not None:
                        prev_motor_l23[_mn] = _ms.region.active_l23.copy()

            # -- Process in topo order (multi-ff supported) --
            encoding = self._encoder.encode(token_str)
            self._propagate_feedforward(topo_order, entry_name, encoding)

            # -- Inter-region signals (after all regions processed) --
            for conn in self._connections:
                if not conn.enabled:
                    continue
                src = self._regions[conn.source].region
                tgt = self._regions[conn.target].region

                if conn.surprise_tracker is not None:
                    n_active = int(src.active_columns.sum())
                    n_bursting = int(src.bursting_columns.sum())
                    burst_rate = n_bursting / max(n_active, 1)
                    modulator = conn.surprise_tracker.update(burst_rate)
                    tgt.surprise_modulator = modulator
                    surprise_modulators[conn.target].append(modulator)

                if conn.role == ConnectionRole.APICAL and tgt.has_apical:
                    src_lamina = src.get_lamina(conn.source_lamina)
                    r_active = int(src.active_columns.sum())
                    r_bursting = int(src.bursting_columns.sum())
                    confidence = 1.0 - (r_bursting / max(r_active, 1))
                    signal = src_lamina.firing_rate * confidence
                    if conn.thalamic_gate is not None:
                        tgt_active = int(tgt.active_columns.sum())
                        tgt_bursting = int(tgt.bursting_columns.sum())
                        tgt_burst_rate = tgt_bursting / max(tgt_active, 1)
                        readiness = conn.thalamic_gate.update(tgt_burst_rate)
                        signal = signal * readiness
                        key = f"{conn.source}->{conn.target}"
                        thalamic_readiness[key].append(readiness)
                    tgt.set_apical_context(signal, source_name=conn.source)

            # -- Per-region bookkeeping (sampled to reduce overhead) --
            is_metric_step = (t % metric_interval == 0) or (t < 100)
            for _name, s in self._regions.items():
                if is_metric_step:
                    s.rep_tracker.observe(
                        token_id, s.region.active_columns, s.region.active_l4
                    )
                if s.diagnostics is not None and is_metric_step:
                    s.diagnostics.step(t, s.region)
                if s.timeline is not None and t % self._timeline_interval == 0:
                    s.timeline.capture(
                        len(s.timeline.frames),
                        s.region,
                        s.region.last_column_drive,
                    )

            # -- Motor metrics + reward --
            for _name, s in self._regions.items():
                if s.motor:
                    assert isinstance(s.region, MotorRegion)
                    motor_region = s.region
                    # observe_token only during active phase (M1 processed)
                    if m1_active:
                        motor_region.observe_token(token_id)
                    if self._total_steps > 0:
                        # BG gating: always step (learns from both phases)
                        gate = 1.0
                        if s.basal_ganglia is not None:
                            precision = (~entry_region.bursting_columns).astype(
                                np.float64
                            )
                            prec_frac = precision.sum() / max(
                                entry_region.n_columns,
                                1,
                            )
                            ctx = self._build_bg_ctx(precision, prec_frac)
                            gate = s.basal_ganglia.step(ctx)
                            if self.force_gate_open:
                                gate = 1.0
                            motor_region.output_scores *= gate
                            metrics[_name].bg_gate_values.append(gate)

                        if m1_active:
                            # M1 processed this step — compute output + metrics
                            pop_id, pop_conf = motor_region.get_population_output()
                            if s.motor_decoder is not None:
                                dec_id, _dec_conf = motor_region.get_decoded_output(
                                    s.motor_decoder,
                                )
                            else:
                                dec_id = -1
                            m_id, m_conf = pop_id, pop_conf
                        else:
                            # M1 idle during input — silent output
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
                        if self._reward_source is not None:
                            m_char = chr(m_id) if spoke and 32 <= m_id < 127 else None
                            reward = self._compute_pluggable_reward(
                                m_char,
                                entry_region,
                            )
                        else:
                            reward = self._compute_turn_reward(
                                spoke,
                                self._in_eom,
                                self._eom_steps,
                                _max_speak_steps,
                            )
                        metrics[_name].motor_rewards.append(reward)

                        # Expose last-step state for interactive use
                        motor_region.last_output = (m_id, m_conf)
                        motor_region.last_gate = gate
                        motor_region.last_reward = reward

                        # BG reward: send computed reward to update gate weights
                        if s.basal_ganglia is not None:
                            if self._reward_source is not None:
                                # Pluggable reward: send directly to BG
                                s.basal_ganglia.reward(reward)
                            else:
                                # Default: turn-taking gate error
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

                        # Apply reward through connections with reward modulators
                        for conn in self._connections:
                            if (
                                conn.source == _name
                                and conn.reward_modulator is not None
                            ):
                                mod = conn.reward_modulator.update(reward)
                                tgt = self._regions[conn.target].region
                                tgt.reward_modulator = mod
                                reward_modulators[conn.target].append(mod)

                        # Efference copy: only during generation (gate forced open)
                        if m_id >= 0 and self.force_gate_open:
                            ef_encoding = self._encoder.encode(
                                chr(m_id) if m_id < 128 else "",
                            )
                            entry_region.set_efference_copy(ef_encoding)

                        # Train motor decoder: previous M1 L2/3 → current token
                        if s.motor_decoder is not None and _name in prev_motor_l23:
                            s.motor_decoder.observe(
                                token_id,
                                prev_motor_l23[_name],
                            )

            # -- Entry metrics (expensive decodes sampled at metric intervals) --
            if is_metric_step and t > 0:
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
                den_predictions = entry_state.dendritic_decoder.decode(
                    entry_region.active_l23
                )
                den_id = den_predictions[0] if den_predictions else -1

                metrics[entry_name].accuracies.append(
                    1.0 if idx_predicted == token_id else 0.0
                )
                metrics[entry_name].synaptic_accuracies.append(
                    1.0 if syn_id == token_id else 0.0
                )
                metrics[entry_name].column_accuracies.append(
                    1.0 if col_id == token_id else 0.0
                )
                metrics[entry_name].dendritic_accuracies.append(
                    1.0 if den_id == token_id else 0.0
                )

                if show_predictions > 0:
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
                    prediction_log.append(
                        (token_str, den_str, idx_str, col_str, syn_str)
                    )

            # BPC: measure prediction quality (sampled at metric intervals)
            if bpc_probe is not None and is_metric_step and t > 0:
                assert entry_state.dendritic_decoder is not None
                bpc_probe.step(
                    token_id,
                    entry_region.active_l23,
                    entry_state.dendritic_decoder,
                )
            if is_metric_step and t > 0:
                centroid_probe.step(token_id, prev_l23)
            centroid_probe.observe(token_id, prev_l23)

            # -- Decoder training (every step — cheap, drives learning) --
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
            for _wd_name, _wd_state in self._regions.items():
                if _wd_state.word_decoder is not None:
                    _wd_state.word_decoder.step(
                        token_str, _wd_state.region.firing_rate_l23
                    )

            self._total_steps += 1

            # -- Logging --
            if (
                t > 0
                and t % log_interval == 0
                and metrics[entry_name].dendritic_accuracies
            ):
                self._log_step(
                    t,
                    start,
                    entry_name,
                    metrics,
                    surprise_modulators,
                    thalamic_readiness,
                    reward_modulators,
                    rolling_window,
                    show_predictions,
                    prediction_log,
                    bpc_probe,
                    centroid_probe,
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

        # Store centroid BPC
        centroid_probe.dialogue_boundary()
        entry_m = metrics[entry_name]
        entry_m.centroid_bpc = centroid_probe.bpc
        entry_m.centroid_bpc_recent = centroid_probe.recent_bpc

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
    # Babbling loop (Stages 2-3)
    # ------------------------------------------------------------------

    def run_babbling(
        self,
        n_steps: int,
        *,
        log_interval: int = 100,
    ) -> dict:
        """Autoregressive babbling loop: M1 drives, hears itself through S1.

        No corpus input. M1 produces a token, that token is encoded and
        fed through S1 (frozen) → S2 (frozen, if connected). Reward is
        computed from S2 pattern stability. M1 learns from the loop.

        Returns dict with babbling metrics.
        """
        if self._entry_name is None:
            raise ValueError("No entry region.")

        entry_name = self._entry_name
        entry_state = self._regions[entry_name]
        entry_region = entry_state.region

        # Find motor region
        motor_state = None
        for _name, s in self._regions.items():
            if s.motor:
                motor_state = s
                break
        if motor_state is None:
            raise ValueError("No motor region for babbling.")
        assert isinstance(motor_state.region, MotorRegion)
        motor_region = motor_state.region

        # Start with a random seed token
        seed_char = " "
        current_token_str = seed_char

        # Adaptive noise: tracks reward EMA. When reward improves,
        # noise decreases (exploit). When reward stagnates or drops,
        # noise increases (explore). Models tonic dopamine regulation.
        noise_floor = 0.05
        noise_ceiling = motor_region.babbling_noise  # initial value as max
        noise = noise_ceiling
        reward_ema = 0.0
        reward_ema_slow = 0.0
        noise_adapt_rate = 0.001  # How fast noise responds to reward changes

        # Metrics
        rewards = []
        gate_values = []
        tokens_produced = []
        unique_tokens = set()

        start = time.monotonic()

        for t in range(n_steps):
            # Adaptive noise: compare fast vs slow reward EMA
            # Fast rising above slow → improving → exploit (reduce noise)
            # Fast falling below slow → stagnating → explore (increase noise)
            # Small |delta| for too long → stagnation → also explore
            if t > 100:
                reward_delta = reward_ema - reward_ema_slow
                # Delta-driven adaptation
                noise -= noise_adapt_rate * reward_delta
                # Stagnation detection: if |delta| is tiny, slowly increase noise
                if abs(reward_delta) < 0.001:
                    noise += noise_adapt_rate * 0.1  # gentle push to explore
                noise = max(noise_floor, min(noise_ceiling, noise))
            motor_region.babbling_noise = noise

            # 1-3. Encode + propagate through hierarchy
            encoding = self._encoder.encode(current_token_str)
            topo_order = self._topo_order()
            self._propagate_feedforward(topo_order, entry_name, encoding)

            # 4. Inter-region signals
            self._propagate_signals()

            # 5-9. Motor output + reward + L5 learning
            pop_id, reward = self._step_motor_reward(entry_region)

            # Update reward EMAs for adaptive noise
            reward_ema = 0.99 * reward_ema + 0.01 * reward  # fast
            reward_ema_slow = 0.999 * reward_ema_slow + 0.001 * reward  # slow

            # Track metrics
            rewards.append(reward)
            m_char = chr(pop_id) if pop_id >= 0 and 32 <= pop_id < 127 else None
            if m_char:
                tokens_produced.append(m_char)
                unique_tokens.add(m_char)

            # 10. Feed M1's output back as next input
            if m_char:
                current_token_str = m_char
            # else: keep previous token (M1 was silent)

            self._total_steps += 1

            # Logging
            if t > 0 and t % log_interval == 0:
                recent_r = rewards[-log_interval:]
                avg_r = sum(recent_r) / len(recent_r)
                recent_tok = tokens_produced[-log_interval:]
                n_unique = len(set(recent_tok))
                burst_pct = float(entry_region.bursting_columns.sum()) / max(
                    entry_region.n_columns, 1
                )
                elapsed = time.monotonic() - start
                # Show last 30 chars M1 produced
                tail = "".join(recent_tok[-30:])
                sample = repr(tail) if tail else "(empty)"
                noise = motor_region.babbling_noise
                print(
                    f"  [babble] t={t:,} "
                    f"r={avg_r:+.3f} "
                    f"noise={noise:.2f} "
                    f"burst={burst_pct:.1%} "
                    f"unique={n_unique} "
                    f"vocab={len(unique_tokens)} "
                    f"out={sample} "
                    f"({elapsed:.1f}s)"
                )

        elapsed = time.monotonic() - start
        return {
            "rewards": rewards,
            "gate_values": gate_values,
            "tokens_produced": tokens_produced,
            "unique_tokens": sorted(unique_tokens),
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    # Interleaved training (listen + babble)
    # ------------------------------------------------------------------

    def run_interleaved(
        self,
        tokens: list[tuple[int, str]],
        n_babble_steps: int,
        *,
        listen_chunk: int = 200,
        babble_chunk: int = 50,
        log_interval: int = 100,
    ) -> dict:
        """Interleaved listening and babbling — like a baby's day.

        Alternates between:
        - Listening: corpus tokens through full hierarchy, M1 observes
        - Babbling: autoregressive M1→S1→M1 loop with reward

        The listening phases continually reinforce L5 token mappings,
        preventing the drift that occurs in pure babbling runs.

        Args:
            tokens: Corpus tokens for listening phases.
            n_babble_steps: Total babble steps (controls run length).
            listen_chunk: Corpus tokens per listening episode.
            babble_chunk: Babble steps per babbling episode.
            log_interval: Steps between log lines.
        """
        if self._entry_name is None:
            raise ValueError("No entry region.")

        entry_name = self._entry_name
        entry_state = self._regions[entry_name]
        entry_region = entry_state.region

        motor_state = None
        motor_region: MotorRegion | None = None
        for _name, s in self._regions.items():
            if s.motor:
                assert isinstance(s.region, MotorRegion)
                motor_state = s
                motor_region = s.region
                break
        if motor_state is None or motor_region is None:
            raise ValueError("No motor region for interleaved training.")

        # Adaptive noise state
        noise_floor = 0.05
        noise_ceiling = motor_region.babbling_noise
        noise = noise_ceiling
        reward_ema = 0.0
        reward_ema_slow = 0.0
        noise_adapt_rate = 0.001

        # Metrics
        rewards = []
        tokens_produced = []
        unique_tokens = set()
        total_listen = 0
        total_babble = 0
        corpus_pos = 0

        start = time.monotonic()

        while total_babble < n_babble_steps:
            # -- LISTEN episode: process corpus tokens --
            chunk_end = min(corpus_pos + listen_chunk, len(tokens))
            if corpus_pos >= len(tokens):
                corpus_pos = 0  # Loop corpus
                chunk_end = min(listen_chunk, len(tokens))

            listen_tokens = tokens[corpus_pos:chunk_end]
            corpus_pos = chunk_end

            # Process listening tokens through standard run loop
            # (M1 learning_enabled means it processes + learns)
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                self.run(listen_tokens, log_interval=999999)
            total_listen += len(listen_tokens)

            # -- BABBLE episode: autoregressive M1 loop --
            seed_char = " "
            current_token_str = seed_char

            for _b in range(babble_chunk):
                if total_babble >= n_babble_steps:
                    break

                # Adaptive noise
                if total_babble > 100:
                    reward_delta = reward_ema - reward_ema_slow
                    noise -= noise_adapt_rate * reward_delta
                    if abs(reward_delta) < 0.001:
                        noise += noise_adapt_rate * 0.1
                    noise = max(noise_floor, min(noise_ceiling, noise))
                motor_region.babbling_noise = noise

                # Encode + propagate + signals + motor reward
                encoding = self._encoder.encode(current_token_str)
                topo_order = self._topo_order()
                self._propagate_feedforward(topo_order, entry_name, encoding)
                self._propagate_signals()
                pop_id, reward = self._step_motor_reward(entry_region)
                m_char = chr(pop_id) if pop_id >= 0 and 32 <= pop_id < 127 else None
                reward_ema = 0.99 * reward_ema + 0.01 * reward
                reward_ema_slow = 0.999 * reward_ema_slow + 0.001 * reward
                rewards.append(reward)
                if m_char:
                    tokens_produced.append(m_char)
                    unique_tokens.add(m_char)
                    current_token_str = m_char

                total_babble += 1
                self._total_steps += 1

            # Logging at end of each babble episode
            if total_babble % log_interval < babble_chunk:
                recent_r = rewards[-babble_chunk:]
                avg_r = sum(recent_r) / max(len(recent_r), 1)
                recent_tok = tokens_produced[-babble_chunk:]
                n_unique = len(set(recent_tok))
                tail = "".join(recent_tok[-30:])
                sample = repr(tail) if tail else "(empty)"
                burst_pct = float(entry_region.bursting_columns.sum()) / max(
                    entry_region.n_columns, 1
                )
                elapsed = time.monotonic() - start
                print(
                    f"  [interleaved] babble={total_babble:,} "
                    f"listen={total_listen:,} "
                    f"r={avg_r:+.3f} "
                    f"noise={noise:.2f} "
                    f"burst={burst_pct:.1%} "
                    f"unique={n_unique} "
                    f"vocab={len(unique_tokens)} "
                    f"out={sample} "
                    f"({elapsed:.1f}s)"
                )

        elapsed = time.monotonic() - start
        return {
            "rewards": rewards,
            "tokens_produced": tokens_produced,
            "unique_tokens": sorted(unique_tokens),
            "total_listen": total_listen,
            "total_babble": total_babble,
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    # Echo training (listen → PFC holds → M1 reproduces)
    # ------------------------------------------------------------------

    def run_echo(
        self,
        tokens: list[tuple[int, str]],
        n_episodes: int,
        *,
        max_word_len: int = 6,
        min_word_len: int = 2,
        log_interval: int = 10,
        batch_reward: bool = False,
        curriculum: bool = False,
        echo_reward_kwargs: dict | None = None,
    ) -> dict:
        """Echo training: hear a word, reproduce it.

        For each episode:
        1. Listen: process chars of a word through hierarchy, PFC open
        2. PFC snapshots goal and closes gate
        3. Speak: M1 produces chars, rewarded for matching heard word
        4. Reset for next word

        Args:
            tokens: Corpus for extracting words.
            n_episodes: Number of listen→echo episodes.
            max_word_len: Max word length to echo (start small).
            min_word_len: Min word length to echo.
            log_interval: Episodes between log lines.
            batch_reward: If True, accumulate reward over episode and
                apply once at end (normalized by word length). Prevents
                per-step weight oscillation.
            curriculum: If True, start with short words (2-3 chars) and
                gradually increase max length as performance improves.
        """
        from step.cortex.pfc import PFCRegion
        from step.cortex.reward import EchoReward

        assert self._entry_name is not None
        entry_name = self._entry_name
        entry_region = self._regions[entry_name].region

        # Find PFC and motor regions
        pfc_region = None
        motor_region: MotorRegion | None = None
        for _name, s in self._regions.items():
            if isinstance(s.region, PFCRegion):
                pfc_region = s.region
            if s.motor:
                assert isinstance(s.region, MotorRegion)
                motor_region = s.region

        if pfc_region is None:
            raise ValueError("No PFC region for echo training.")
        if motor_region is None:
            raise ValueError("No motor region for echo training.")

        # Extract words from corpus, grouped by length for curriculum
        words_by_len: dict[int, list[list[str]]] = {}
        current: list[str] = []
        for token_id, ch in tokens:
            if token_id < 0 or ch in (" ", ".", ",", "!", "?", "'", "-", ""):
                wlen = len(current)
                if min_word_len <= wlen <= max_word_len:
                    words_by_len.setdefault(wlen, []).append(current[:])
                current.clear()
            else:
                current.append(ch)
        all_words = [w for ws in words_by_len.values() for w in ws]
        if not all_words:
            raise ValueError("No words found in corpus.")

        echo_reward = EchoReward(**(echo_reward_kwargs or {}))
        old_reward = self._reward_source
        self._reward_source = echo_reward

        # Curriculum state: start with shortest words, advance when stable
        if curriculum:
            available_lens = sorted(words_by_len.keys())
            curr_max_len = available_lens[0] if available_lens else max_word_len
            curriculum_window: list[float] = []
            curriculum_threshold = 0.25  # advance when avg match > 25%
        else:
            curr_max_len = max_word_len

        # Metrics
        rewards: list[float] = []
        matches: list[float] = []
        start = time.monotonic()

        for ep in range(n_episodes):
            # Pick a word (respecting curriculum length limit)
            if curriculum:
                eligible = [
                    w
                    for ws in words_by_len.values()
                    for w in ws
                    if len(w) <= curr_max_len
                ]
                word = eligible[ep % len(eligible)]
            else:
                word = all_words[ep % len(all_words)]

            # -- LISTEN phase: PFC gate open, process word chars --
            pfc_region.gate_open = True
            echo_reward.reset()

            for ch in word:
                encoding = self._encoder.encode(ch)
                topo_order = self._topo_order()
                self._propagate_feedforward(topo_order, entry_name, encoding)
                self._propagate_signals()
                echo_reward.hear(ch)

            # PFC snapshots and closes gate
            pfc_region.snapshot_goal()
            pfc_region.gate_open = False

            # -- SPEAK phase: M1 tries to reproduce --
            echo_reward.start_speak()
            self.force_gate_open = True  # M1 active
            motor_region.babbling_noise = 0.2  # Mostly learned, some explore
            episode_reward = 0.0
            step_rewards: list[float] = []
            produced: list[str] = []

            for _step_i in range(len(word) + 2):
                # PFC→M2→M1 flows through normal ff propagation
                # (PFC gate closed, so it maintains goal pattern;
                #  its firing_rate_l23 feeds M2 via PFC→M2 ff connection)

                # Feed last produced char (or space as seed)
                seed = produced[-1] if produced else " "
                encoding = self._encoder.encode(seed)
                topo_order = self._topo_order()
                self._propagate_feedforward(topo_order, entry_name, encoding)
                self._propagate_signals()

                if batch_reward:
                    # Accumulate reward but don't apply per-step
                    pop_id, reward = self._step_motor_reward_no_apply(entry_region)
                    step_rewards.append(reward)
                else:
                    pop_id, reward = self._step_motor_reward(entry_region)

                episode_reward += reward

                m_char = chr(pop_id) if pop_id >= 0 and 32 <= pop_id < 127 else None
                if m_char:
                    produced.append(m_char)

                self._total_steps += 1

            self.force_gate_open = False

            # Batch reward: apply mean reward once at episode end
            if batch_reward and step_rewards:
                batch_r = sum(step_rewards) / len(step_rewards)
                if hasattr(motor_region, "apply_reward"):
                    motor_region.apply_reward(batch_r)
                if hasattr(pfc_region, "apply_reward"):
                    pfc_region.apply_reward(batch_r)

            # Score: how many chars matched?
            n_match = sum(
                1
                for i, ch in enumerate(produced[: len(word)])
                if i < len(word) and ch == word[i]
            )
            match_rate = n_match / max(len(word), 1)
            rewards.append(episode_reward)
            matches.append(match_rate)

            # Curriculum: advance word length when performance is stable
            if curriculum:
                curriculum_window.append(match_rate)
                if len(curriculum_window) > 50:
                    curriculum_window.pop(0)
                if (
                    len(curriculum_window) >= 50
                    and sum(curriculum_window) / len(curriculum_window)
                    > curriculum_threshold
                    and curr_max_len < max_word_len
                ):
                    curr_max_len = min(curr_max_len + 1, max_word_len)
                    curriculum_window.clear()
                    print(f"  [echo] curriculum: advancing to max_len={curr_max_len}")

            # Route reward to PFC via three-factor consolidation.
            # PFC's eligibility traces captured which S2+S3→PFC
            # mappings were active during the listen phase. Reward
            # consolidates: good echo → strengthen those mappings,
            # bad echo → weaken them. No replay needed.
            if not batch_reward:
                avg_r = episode_reward / max(len(word), 1)
                if hasattr(pfc_region, "apply_reward"):
                    pfc_region.apply_reward(avg_r)

            # Reset for next episode
            for s in self._regions.values():
                s.region.reset_working_memory()

            # Logging
            if (ep + 1) % log_interval == 0:
                recent_r = rewards[-log_interval:]
                recent_m = matches[-log_interval:]
                avg_r = sum(recent_r) / len(recent_r)
                avg_m = sum(recent_m) / len(recent_m)
                heard = "".join(word)
                said = "".join(produced[: len(word)])
                elapsed = time.monotonic() - start
                extra = f" maxlen={curr_max_len}" if curriculum else ""
                print(
                    f"  [echo] ep={ep + 1:,} "
                    f"r={avg_r:+.3f} "
                    f"match={avg_m:.0%} "
                    f"heard={heard!r} "
                    f"said={said!r}"
                    f"{extra} "
                    f"({elapsed:.1f}s)"
                )

        self._reward_source = old_reward
        elapsed = time.monotonic() - start

        return {
            "rewards": rewards,
            "matches": matches,
            "avg_match": sum(matches) / max(len(matches), 1),
            "echo_summary": echo_reward.summary(),
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    # Dialogue training (listen → PFC holds → M1 responds)
    # ------------------------------------------------------------------

    def run_dialogue(
        self,
        tokens: list[tuple[int, str]],
        n_turns: int,
        *,
        max_utterance_len: int = 30,
        max_response_len: int = 15,
        log_interval: int = 50,
    ) -> dict:
        """Dialogue training: alternate listening and responding.

        Each turn:
        1. Listen to a corpus utterance (up to boundary/max_len)
        2. PFC gate closes — holds context from what was heard
        3. M1 produces response with PFC goal drive
        4. Reward: echo match on first word of utterance (bootstrap)
           + curiosity for natural transitions
        5. PFC learning modulated by reward
        6. Reset for next turn

        This is the first structured conversational training — the
        precursor to actual dialogue.
        """
        from step.cortex.pfc import PFCRegion
        from step.cortex.reward import EchoReward

        assert self._entry_name is not None
        entry_name = self._entry_name
        entry_region = self._regions[entry_name].region

        pfc_region = None
        motor_region: MotorRegion | None = None
        for _name, s in self._regions.items():
            if isinstance(s.region, PFCRegion):
                pfc_region = s.region
            if s.motor:
                assert isinstance(s.region, MotorRegion)
                motor_region = s.region

        if pfc_region is None or motor_region is None:
            raise ValueError("Need PFC and motor regions for dialogue.")

        if motor_region._goal_weights is None:
            motor_region.init_goal_drive(pfc_region.n_l23_total)

        echo_reward = EchoReward()
        old_reward = self._reward_source
        self._reward_source = echo_reward

        # Extract utterances from corpus (split on boundaries)
        utterances = []
        current = []
        for token_id, ch in tokens:
            if token_id < 0:  # STORY_BOUNDARY
                if current:
                    utterances.append(current[:])
                    current.clear()
            else:
                current.append(ch)
                if len(current) >= max_utterance_len or ch in (".", "!", "?"):
                    utterances.append(current[:])
                    current.clear()
        if not utterances:
            raise ValueError("No utterances found.")

        rewards = []
        matches = []
        start = time.monotonic()

        for turn in range(n_turns):
            utt = utterances[turn % len(utterances)]
            echo_reward.reset()

            # -- LISTEN phase --
            pfc_region.gate_open = True
            heard_word = []  # Track first word for echo reward

            for ch in utt:
                encoding = self._encoder.encode(ch)
                topo_order = self._topo_order()
                self._propagate_feedforward(topo_order, entry_name, encoding)
                self._propagate_signals()

                # Track first word
                if ch in (" ", ".", ",", "!", "?", "-"):
                    if not heard_word:
                        pass  # Haven't started a word yet
                elif not any(c in (" ", ".", "!", "?") for c in heard_word):
                    heard_word.append(ch)

                echo_reward.hear(ch)

            # PFC holds context
            pfc_region.snapshot_goal()
            pfc_region.gate_open = False

            # -- RESPOND phase --
            echo_reward.start_speak()
            self.force_gate_open = True
            motor_region.babbling_noise = 0.2
            episode_reward = 0.0
            produced = []

            for _step_i in range(max_response_len):
                motor_region.set_goal_drive(pfc_region.firing_rate_l23)

                seed = produced[-1] if produced else " "
                encoding = self._encoder.encode(seed)
                topo_order = self._topo_order()
                self._propagate_feedforward(topo_order, entry_name, encoding)
                self._propagate_signals()

                pop_id, reward = self._step_motor_reward(entry_region)
                episode_reward += reward

                m_char = chr(pop_id) if pop_id >= 0 and 32 <= pop_id < 127 else None
                if m_char:
                    produced.append(m_char)

                self._total_steps += 1

            self.force_gate_open = False

            # Score against first word of utterance
            first_word = "".join(c for c in heard_word[:6] if c.isalpha())
            response = "".join(produced[: len(first_word)])
            n_match = sum(
                1
                for i, ch in enumerate(response)
                if i < len(first_word) and ch == first_word[i]
            )
            match_rate = n_match / max(len(first_word), 1)
            rewards.append(episode_reward)
            matches.append(match_rate)

            # PFC three-factor consolidation (no replay needed)
            avg_r = episode_reward / max(max_response_len, 1)
            if hasattr(pfc_region, "apply_reward"):
                pfc_region.apply_reward(avg_r)

            # Reset
            for s in self._regions.values():
                s.region.reset_working_memory()

            # Logging
            if (turn + 1) % log_interval == 0:
                recent_r = rewards[-log_interval:]
                recent_m = matches[-log_interval:]
                avg_r = sum(recent_r) / len(recent_r)
                avg_m = sum(recent_m) / len(recent_m)
                heard = "".join(utt[:20])
                said = "".join(produced[:20])
                elapsed = time.monotonic() - start
                print(
                    f"  [dialogue] turn={turn + 1:,} "
                    f"r={avg_r:+.3f} "
                    f"match={avg_m:.0%} "
                    f"heard={heard!r} "
                    f"said={said!r} "
                    f"({elapsed:.1f}s)"
                )

        self._reward_source = old_reward
        elapsed = time.monotonic() - start

        return {
            "rewards": rewards,
            "matches": matches,
            "avg_match": sum(matches) / max(len(matches), 1),
            "elapsed_seconds": elapsed,
        }

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
                # l23_lateral_weights removed — L2/3 lateral uses segments only
                "fb_seg_indices": r.fb_seg_indices,
                "fb_seg_perm": r.fb_seg_perm,
                "lat_seg_indices": r.lat_seg_indices,
                "lat_seg_perm": r.lat_seg_perm,
                "l23_seg_indices": r.l23_seg_indices,
                "l23_seg_perm": r.l23_seg_perm,
                "l5_seg_indices": r.l5_seg_indices,
                "l5_seg_perm": r.l5_seg_perm,
            }
            # Apical per-source state (weights or segments)
            if r._apical_sources:
                apical_data = {}
                for name, src in r._apical_sources.items():
                    if "weights" in src:
                        apical_data[name] = {"weights": src["weights"].copy()}
                    elif "seg_indices" in src:
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
            # Old checkpoints may have l23_lateral_weights — skip it.
            # L2/3 lateral now uses segments only.
            r.fb_seg_indices[:] = region_data["fb_seg_indices"]
            r.fb_seg_perm[:] = region_data["fb_seg_perm"]
            r.lat_seg_indices[:] = region_data["lat_seg_indices"]
            r.lat_seg_perm[:] = region_data["lat_seg_perm"]
            r.l23_seg_indices[:] = region_data["l23_seg_indices"]
            r.l23_seg_perm[:] = region_data["l23_seg_perm"]
            if "l5_seg_indices" in region_data:
                r.l5_seg_indices[:] = region_data["l5_seg_indices"]
                r.l5_seg_perm[:] = region_data["l5_seg_perm"]

            # Load apical state (per-source weights or segments)
            if "apical_sources" in region_data:
                for src_name, saved in region_data["apical_sources"].items():
                    if src_name not in r._apical_sources:
                        continue
                    src = r._apical_sources[src_name]
                    if isinstance(saved, dict):
                        # New format: dict with weights or seg_indices/seg_perm
                        if "weights" in saved and "weights" in src:
                            src["weights"][:] = saved["weights"]
                        if "seg_indices" in saved and "seg_indices" in src:
                            src["seg_indices"][:] = saved["seg_indices"]
                            src["seg_perm"][:] = saved["seg_perm"]
                    else:
                        # Legacy format: bare weight array
                        if "weights" in src:
                            src["weights"][:] = saved
            elif (
                "apical_gain_weights" in region_data
                and r._apical_gain_weights is not None
            ):
                # Legacy: single-source checkpoint
                r._apical_gain_weights[:] = region_data["apical_gain_weights"]

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

    def _propagate_feedforward(self, topo_order, entry_name, encoding=None):
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
            elif s.motor and not (
                self._in_eom or self.force_gate_open or s.region.learning_enabled
            ):
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
                r_active = max(int(src.active_columns.sum()), 1)
                r_bursting = int(src.bursting_columns.sum())
                confidence = 1.0 - (r_bursting / r_active)
                signal = src_lamina.firing_rate * confidence
                if conn.thalamic_gate is not None:
                    tgt_active = max(int(tgt.active_columns.sum()), 1)
                    tgt_bursting = int(tgt.bursting_columns.sum())
                    tgt_burst_rate = tgt_bursting / tgt_active
                    conn.thalamic_gate.update(tgt_burst_rate)
                tgt.set_apical_context(signal, source_name=conn.source)

    def _build_bg_ctx(self, precision: np.ndarray, prec_frac: float) -> np.ndarray:
        """Build BG context vector using a pre-allocated buffer."""
        n = precision.shape[0]
        if self._bg_ctx_buffer is None or self._bg_ctx_buffer.shape[0] != n + 1:
            self._bg_ctx_buffer = np.empty(n + 1, dtype=np.float64)
        self._bg_ctx_buffer[:n] = precision
        self._bg_ctx_buffer[n] = prec_frac
        return self._bg_ctx_buffer

    def _step_motor_reward(self, entry_region) -> tuple[int, float]:
        """Process motor output, BG gating, and reward. Returns (token_id, reward)."""
        for _name, s in self._regions.items():
            if not s.motor:
                continue
            assert isinstance(s.region, MotorRegion)
            motor_region = s.region
            pop_id, _pop_conf = motor_region.get_population_output()

            # BG gating
            gate = 1.0
            if s.basal_ganglia is not None:
                precision = (~entry_region.bursting_columns).astype(np.float64)
                prec_frac = precision.sum() / max(entry_region.n_columns, 1)
                ctx = self._build_bg_ctx(precision, prec_frac)
                gate = s.basal_ganglia.step(ctx)
                if self.force_gate_open:
                    gate = 1.0
                motor_region.output_scores *= gate

            # Reward
            m_char = chr(pop_id) if pop_id >= 0 and 32 <= pop_id < 127 else None
            if self._reward_source is not None:
                reward = self._compute_pluggable_reward(m_char, entry_region)
            else:
                reward = 0.0

            # Apply reward to BG + M1
            if s.basal_ganglia is not None:
                if self._reward_source is not None:
                    s.basal_ganglia.reward(reward)
                else:
                    gate_target = 1.0 if self._in_eom else 0.0
                    gate_error = gate_target - s.basal_ganglia.gate_value
                    s.basal_ganglia.reward(gate_error)
            if hasattr(motor_region, "apply_reward"):
                motor_region.apply_reward(reward)

            # Update L5 output weights
            if pop_id >= 0:
                motor_region.observe_token(pop_id)

            return pop_id, reward

        return -1, 0.0

    def _step_motor_reward_no_apply(self, entry_region) -> tuple[int, float]:
        """Like _step_motor_reward but skip weight consolidation.

        Used for batch reward: accumulate rewards over an episode, then
        call motor_region.apply_reward() once with the mean.
        """
        for _name, s in self._regions.items():
            if not s.motor:
                continue
            assert isinstance(s.region, MotorRegion)
            motor_region = s.region
            pop_id, _pop_conf = motor_region.get_population_output()

            # BG gating (still applied per-step for output scores)
            gate = 1.0
            if s.basal_ganglia is not None:
                precision = (~entry_region.bursting_columns).astype(np.float64)
                prec_frac = precision.sum() / max(entry_region.n_columns, 1)
                ctx = self._build_bg_ctx(precision, prec_frac)
                gate = s.basal_ganglia.step(ctx)
                if self.force_gate_open:
                    gate = 1.0
                motor_region.output_scores *= gate

            # Compute reward but don't apply
            m_char = chr(pop_id) if pop_id >= 0 and 32 <= pop_id < 127 else None
            if self._reward_source is not None:
                reward = self._compute_pluggable_reward(m_char, entry_region)
            else:
                reward = 0.0

            # BG still gets per-step reward (gating should be responsive)
            if s.basal_ganglia is not None and self._reward_source is not None:
                s.basal_ganglia.reward(reward)

            # Update L5 output weights (observation is not reward-dependent)
            if pop_id >= 0:
                motor_region.observe_token(pop_id)

            return pop_id, reward

        return -1, 0.0

    def _compute_pluggable_reward(
        self,
        char: str | None,
        entry_region,
    ) -> float:
        """Compute reward from pluggable source with appropriate context."""
        from step.cortex.reward import (
            CaregiverReward,
            CuriosityReward,
            EchoReward,
        )

        assert self._reward_source is not None
        reward_types = (CuriosityReward, CaregiverReward, EchoReward)
        if isinstance(self._reward_source, reward_types):
            # Curiosity/Caregiver: pass S1 burst fraction
            n_active = max(int(entry_region.active_columns.sum()), 1)
            n_bursting = int(entry_region.bursting_columns.sum())
            burst_frac = n_bursting / n_active
            return self._reward_source.step(char, burst_frac)
        else:
            # Generic reward source: pass S2 columns
            s2_cols = np.zeros(0)
            for _rn, _rs in self._regions.items():
                if _rn == "S2":
                    s2_cols = _rs.region.active_columns
                    break
            return self._reward_source.step(char, s2_cols)

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
        lamina = src.get_lamina(conn.source_lamina)
        signal = lamina.firing_rate.copy()

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

    def _topo_order(self) -> list[str]:
        """Return cached topological order. Auto-finalizes if needed."""
        if not self._finalized:
            self.finalize()
        assert self._topo_cache is not None
        return self._topo_cache

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
            import math

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
