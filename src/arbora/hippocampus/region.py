"""HippocampalRegion: composite that wires EC → DG → CA3 → CA1.

The single hippocampal object the Circuit sees. Internally owns
`EntorhinalCortex`, `DentateGyrus`, `CA3`, `CA1` and runs them in a
fixed pipeline on every `process()` call.

Pipeline
--------
Given a cortical input of shape (input_dim,):

1. ``ec_pat = ec.forward(encoding)`` — sparse binary EC code
2. ``dg_pat = dg.forward(ec_pat)`` — sparse binary DG code (pattern-separated)
3. ``bound = ca3.encode(dg_pat)`` — mossy-driven activation + one-shot
   LTP on lateral weights
4. ``completed = ca3.retrieve(bound, n_iter=...)`` — recurrent pattern
   completion over the attractor
5. ``output, match = ca1.forward(completed, ec_pat)`` — dual-input
   comparator; match is cosine similarity of the two drives
6. ``cortex_out = ec.reverse(output)`` — back to cortical dim

The encode-then-retrieve sequence matters: for fresh inputs the retrieve
step simply confirms the mossy-driven pattern, but for inputs whose DG
code overlaps a previously-stored memory the retrieve step pulls the
activation toward the stored attractor (pattern completion).

Symmetric I/O
-------------
For v1 HC is symmetric: ``input_dim == output_dim``. EC.reverse projects
back to the original cortical dimension. Asymmetric wiring (e.g. S1 dim
!= M1 dim) will need a separate projection layer or an EC variant with
independent reverse dim; defer until ARB-117 actually hits that case.

v1 limitations
--------------
- Salience gating is not wired. Every `process()` call encodes (LTP on
  CA3 lateral weights). Without episodic resets this fills CA3 with
  noise on long runs. v1.1 will gate encoding on CuriosityReward RPE.
- No learned CA1 or EC representations.
- No replay / systems consolidation — cortical drift invalidates stored
  CA3 keys over time. Call `reset_memory()` at task boundaries as a
  crude mitigation (ARB-120 is the real fix).
"""

from __future__ import annotations

import numpy as np

from arbora.hippocampus.ca1 import CA1
from arbora.hippocampus.ca3 import CA3
from arbora.hippocampus.dentate_gyrus import DentateGyrus
from arbora.hippocampus.entorhinal_cortex import EntorhinalCortex
from arbora.neuron_group import NeuronGroup


class HippocampalRegion:
    """Composite hippocampal region; satisfies the `Region` protocol.

    Parameters
    ----------
    input_dim : int
        Cortical input dimension. Output dimension equals this (symmetric).
    ec_dim : int
        EC-space dimension. Default 1000.
    dg_dim : int
        DG granule cell count. Default 4000 (~4x ec_dim, matches rodent
        overprovisioning ratio).
    ca3_dim : int
        CA3 attractor dimension. Default 1000.
    ec_sparsity : float
        Fraction of EC units active after k-WTA. Default 0.02.
    dg_k : int | None
        Active DG units. Defaults to 2% of `dg_dim` when not given.
    ca3_k_active : float
        Fraction of CA3 units active after k-WTA. Default 0.02.
    ca3_mossy_sparsity : float
        Fraction of DG units that project to each CA3 unit. Default 0.02.
    ca3_mossy_weight : float
        Per-contact strength of the mossy detonator. Default 2.0.
    ca3_learning_rate : float
        CA3 lateral LTP increment per encode. Default 0.5.
    retrieval_iterations : int
        Recurrent iterations per `process()` call. Default 3.
    seed : int
        Base seed. Each internal layer gets a distinct offset so their
        random matrices are independent.

    Attributes
    ----------
    ec, dg, ca3, ca1
        The internal layer instances. Directly accessible for probes
        and advanced tuning (e.g. `region.ca3.learning_rate = 0.3`).
    last_match : float
        Cosine similarity reported by CA1 on the most recent process()
        call. Updated every step. Reserved for the v1.1 salience gate.
    last_ec_pattern : np.ndarray, shape (ec_dim,), dtype=bool
        EC forward output from the most recent `process()` call.
        Exposed for probes that want to observe HC's intermediate
        state without re-running the pipeline. Zeroed before any call.
    last_dg_pattern : np.ndarray, shape (dg_dim,), dtype=bool
        DG output from the most recent `process()` call. Same role as
        `last_ec_pattern` — mechanistic observability for probes.
    """

    INPUT_ID = "hc_in"
    OUTPUT_ID = "hc_out"

    def __init__(
        self,
        input_dim: int,
        *,
        ec_dim: int = 1000,
        dg_dim: int = 4000,
        ca3_dim: int = 1000,
        ec_sparsity: float = 0.02,
        dg_k: int | None = None,
        ca3_k_active: float = 0.02,
        ca3_mossy_sparsity: float = 0.02,
        ca3_mossy_weight: float = 2.0,
        ca3_learning_rate: float = 0.5,
        retrieval_iterations: int = 3,
        seed: int = 0,
    ):
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive; got {input_dim}")
        if retrieval_iterations < 0:
            raise ValueError(
                f"retrieval_iterations must be non-negative; got {retrieval_iterations}"
            )

        self._input_dim = input_dim
        self._retrieval_iterations = retrieval_iterations

        self.ec = EntorhinalCortex(
            input_dim=input_dim,
            output_dim=ec_dim,
            sparsity=ec_sparsity,
            seed=seed,
        )
        if dg_k is None:
            dg_k = max(1, round(dg_dim * 0.02))
        self.dg = DentateGyrus(
            input_dim=ec_dim,
            output_dim=dg_dim,
            k=dg_k,
            seed=seed + 1,
        )
        self.ca3 = CA3(
            dim=ca3_dim,
            dg_dim=dg_dim,
            mossy_sparsity=ca3_mossy_sparsity,
            mossy_weight=ca3_mossy_weight,
            k_active=ca3_k_active,
            learning_rate=ca3_learning_rate,
            seed=seed + 2,
        )
        self.ca1 = CA1(
            ca3_dim=ca3_dim,
            ec_direct_dim=ec_dim,
            output_dim=ec_dim,
            seed=seed + 3,
        )

        self._input_group = NeuronGroup(
            n_neurons=input_dim, group_id=self.INPUT_ID, region=self
        )
        self._output_group = NeuronGroup(
            n_neurons=input_dim, group_id=self.OUTPUT_ID, region=self
        )

        self.last_match: float = 0.0
        # Observable intermediate state for probes. CA3 already exposes
        # `.state` and `.lateral_weights`, so no duplicate buffers for it.
        self.last_ec_pattern: np.ndarray = np.zeros(ec_dim, dtype=np.bool_)
        self.last_dg_pattern: np.ndarray = np.zeros(dg_dim, dtype=np.bool_)

    # ------------------------------------------------------------------
    # Region protocol
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def input_port(self) -> NeuronGroup:
        return self._input_group

    @property
    def output_port(self) -> NeuronGroup:
        return self._output_group

    def process(self, encoding: np.ndarray, **kwargs) -> np.ndarray:
        """Run EC → DG → CA3 → CA1 → EC.reverse and emit a cortex-dim vector.

        Side effects: writes `output_port.firing_rate`, updates `last_match`.
        """
        ec_pat = self.ec.forward(encoding)
        dg_pat = self.dg.forward(ec_pat)
        bound = self.ca3.encode(dg_pat)
        completed = self.ca3.retrieve(bound, n_iter=self._retrieval_iterations)
        output, match = self.ca1.forward(completed, ec_pat)
        cortex_out = self.ec.reverse(output)

        self._output_group.firing_rate[:] = cortex_out
        self.last_match = float(match)
        self.last_ec_pattern = ec_pat
        self.last_dg_pattern = dg_pat
        return cortex_out

    def apply_reward(self, reward: float) -> None:
        """No-op. HC learning in v1 is not reward-modulated."""

    def reset_working_memory(self) -> None:
        """Clear transient state. Preserves CA3 lateral weights (the memory)."""
        self.ca3.reset()
        self._input_group.clear_modulation()
        self._output_group.firing_rate[:] = 0.0
        self.last_match = 0.0
        self.last_ec_pattern[:] = False
        self.last_dg_pattern[:] = False

    def reset_memory(self) -> None:
        """Clear both transient state and CA3 lateral weights.

        Episodic memory reset. Call at task boundaries when cortical
        drift has invalidated stored CA3 keys — see the module docstring
        on v1 limitations.
        """
        self.ca3.reset_memory()
        self.reset_working_memory()
