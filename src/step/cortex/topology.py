"""Topology: declarative region wiring that replaces boilerplate run loops.

Build a topology by adding regions and connections, then call run() once.
Supports single-region, two-region hierarchy, and arbitrary DAGs.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.lamina import LaminaID
from step.cortex.modulators import RewardModulator, SurpriseTracker, ThalamicGate
from step.cortex.motor import MotorRegion
from step.cortex.region import CorticalRegion
from step.cortex.topology_runner import TopologyRunner
from step.cortex.topology_types import (
    Connection,
    ConnectionRole,
    CortexResult,
    Encoder,
    RunMetrics,
    _RegionState,
)
from step.decoders import DendriticDecoder, InvertedIndexDecoder, SynapticDecoder
from step.probes.diagnostics import CortexDiagnostics
from step.probes.representation import RepresentationTracker
from step.probes.timeline import Timeline

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


class Topology(TopologyRunner):
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

            if isinstance(r, MotorRegion):
                region_data["output_weights"] = r.output_weights
                region_data["output_mask"] = r.output_mask
                region_data["ff_eligibility"] = r._ff_eligibility
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

            if isinstance(r, MotorRegion):
                if "output_weights" in region_data:
                    r.output_weights[:] = region_data["output_weights"]
                    r.output_mask[:] = region_data["output_mask"]
                if "ff_eligibility" in region_data:
                    r._ff_eligibility[:] = region_data["ff_eligibility"]
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
