"""Training stage configuration for developmental learning.

Each stage defines which regions learn, which connections are active,
and what reward signals are used. Stages are applied to a Circuit
via their configure function.

Stages model infant development:
  1. Sensory:  T1->T2->T3 representation learning
  2. Babbling: M1->T1->M1 motor exploration (T1 frozen)
  3. Guided:   M1+BG+T2 word-level RL
  4. Imitation: T1->T2->M2->M1 echolalia
  5. Generation: PFC->M2->M1 goal-directed RL
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

from arbora.cortex.circuit import ConnectionRole


@dataclass
class TrainingStage:
    """Metadata and configure callback for a training stage.

    The ``configure`` callable applies freeze/unfreeze, connection
    enable/disable, motor noise, and reward source settings to a
    Circuit.
    """

    name: str
    description: str

    # Tokens to train for this stage
    n_tokens: int = 100_000

    # Regions where learning is enabled (for display / metadata)
    learning_regions: list[str] = field(default_factory=list)

    # Checkpoint to load before this stage (None = start fresh or continue)
    load_checkpoint: str | None = None

    # Checkpoint to save after this stage
    save_checkpoint: str | None = None

    # Motor babbling noise (0.0 = off, 1.0 = pure random)
    exploration_noise: float = 0.0

    # Force M1 active every step (not just EOM phase)
    force_motor_active: bool = False

    # Reward source: "turn_taking" (default), "curiosity", or "caregiver"
    reward_source: str = "turn_taking"

    # Stage configuration function: (circuit) -> None
    configure: Callable | None = None


# ------------------------------------------------------------------
# Configure helpers (shared logic)
# ------------------------------------------------------------------


def _apply_learning_regions(circuit, regions: list[str]) -> None:
    """Freeze/unfreeze regions based on the learning list."""
    if not regions:
        return
    for name in circuit._regions:
        if name in regions:
            circuit.unfreeze_region(name)
        else:
            circuit.freeze_region(name)


def _apply_motor_settings(
    circuit, exploration_noise: float, force_motor_active: bool
) -> None:
    """Configure motor babbling noise.

    force_motor_active is stored on TrainingStage as metadata —
    the harness reads it and sets agent.force_gate_open.
    """
    for _name, state in circuit._regions.items():
        if state.motor:
            state.region.exploration_noise = exploration_noise


def _apply_reward_source(circuit, reward_source: str) -> None:
    """Set the reward source on the circuit."""
    if reward_source == "caregiver":
        from examples.chat.reward import CaregiverReward

        circuit.set_reward_source(CaregiverReward())
    elif reward_source == "curiosity":
        from examples.chat.reward import CuriosityReward

        circuit.set_reward_source(CuriosityReward())
    else:
        circuit.set_reward_source(None)  # default turn-taking


# ------------------------------------------------------------------
# Stage configure functions
# ------------------------------------------------------------------


def configure_sensory(circuit) -> None:
    """Sensory stage: all regions learn, motor pathway listens.

    Enables the full sensory hierarchy and motor listening pathway.
    Disables motor monitoring apical (M1->M2, M2->PFC) and
    M1->T1 apical (motor shouldn't influence sensory during learning).
    """
    _apply_learning_regions(circuit, ["T1", "T2", "T3", "M1", "M2", "PFC"])
    _apply_motor_settings(circuit, exploration_noise=0.0, force_motor_active=False)
    _apply_reward_source(circuit, "turn_taking")

    # Sensory feedforward: on
    circuit.enable_connection("T1", "T2", ConnectionRole.FEEDFORWARD)
    circuit.enable_connection("T2", "T3", ConnectionRole.FEEDFORWARD)

    # Apical feedback (sensory top-down): on
    circuit.enable_connection("T2", "T1", ConnectionRole.APICAL)
    circuit.enable_connection("T3", "T2", ConnectionRole.APICAL)

    # Motor pathway listening: T2->M2->M1
    circuit.enable_connection("T2", "M2", ConnectionRole.FEEDFORWARD)
    circuit.enable_connection("M2", "M1", ConnectionRole.FEEDFORWARD)

    # PFC listening: T2->PFC, T3->PFC, PFC->M2
    circuit.enable_connection("T2", "PFC", ConnectionRole.FEEDFORWARD)
    circuit.enable_connection("T3", "PFC", ConnectionRole.FEEDFORWARD)
    circuit.enable_connection("PFC", "M2", ConnectionRole.FEEDFORWARD)

    # Cross-hierarchy apical: T1->M1 on (surprise carries to motor)
    circuit.enable_connection("T1", "M1", ConnectionRole.APICAL)

    # Motor->sensory apical: off during listening
    _try_disable(circuit, "M1", "T1", ConnectionRole.APICAL)

    # Motor monitoring apical: off during listening
    _try_disable(circuit, "M1", "M2", ConnectionRole.APICAL)
    _try_disable(circuit, "M2", "PFC", ConnectionRole.APICAL)


def configure_babbling(circuit) -> None:
    """Babbling stage: interleaved listening + babbling with caregiver reward.

    Same connection pattern as sensory (full hierarchy for listening),
    but with motor babbling noise and caregiver reward enabled.
    """
    # Connection pattern is the same as sensory
    configure_sensory(circuit)

    # Override motor and reward settings for babbling
    _apply_learning_regions(circuit, ["T1", "T2", "T3", "M1", "M2", "PFC"])
    _apply_motor_settings(circuit, exploration_noise=0.5, force_motor_active=True)
    _apply_reward_source(circuit, "caregiver")


def configure_guided_babbling(circuit) -> None:
    """Guided babbling: T1->M1 + T1->T2 (for word reward), apical on.

    Only M1 learns. Motor monitoring off. T3 pathway disabled.
    """
    _apply_learning_regions(circuit, ["M1"])
    _apply_motor_settings(circuit, exploration_noise=0.5, force_motor_active=True)
    _apply_reward_source(circuit, "caregiver")

    # T1->M1 feedforward: on
    _try_enable(circuit, "T1", "M1", ConnectionRole.FEEDFORWARD)

    # T1->T2 feedforward: on (T2 provides reward signal)
    circuit.enable_connection("T1", "T2", ConnectionRole.FEEDFORWARD)

    # T2->T1 apical: on (word context helps)
    circuit.enable_connection("T2", "T1", ConnectionRole.APICAL)

    # T3 pathway: off
    _try_disable(circuit, "T2", "T3", ConnectionRole.FEEDFORWARD)
    _try_disable(circuit, "T3", "T2", ConnectionRole.APICAL)

    # M1->T1 apical: off
    _try_disable(circuit, "M1", "T1", ConnectionRole.APICAL)


# ------------------------------------------------------------------
# Helpers for connections that may not exist in all topologies
# ------------------------------------------------------------------


def _try_disable(circuit, source: str, target: str, role: ConnectionRole) -> None:
    """Disable a connection, silently ignoring if it doesn't exist."""
    with contextlib.suppress(ValueError):
        circuit.disable_connection(source, target, role)


def _try_enable(circuit, source: str, target: str, role: ConnectionRole) -> None:
    """Enable a connection, silently ignoring if it doesn't exist."""
    with contextlib.suppress(ValueError):
        circuit.enable_connection(source, target, role)


# ------------------------------------------------------------------
# Predefined stages
# ------------------------------------------------------------------

SENSORY_STAGE = TrainingStage(
    name="sensory",
    description="Sensory learning + motor pathway listening",
    n_tokens=1_000_000,
    learning_regions=["T1", "T2", "T3", "M1", "M2", "PFC"],
    save_checkpoint="stage1_sensory",
    configure=configure_sensory,
)

BABBLING_STAGE = TrainingStage(
    name="babbling",
    description="Interleaved listening + babbling with caregiver reward",
    n_tokens=200_000,
    learning_regions=["T1", "T2", "T3", "M1", "M2", "PFC"],
    load_checkpoint="stage1_sensory",
    save_checkpoint="stage2_babbling",
    exploration_noise=0.5,
    force_motor_active=True,
    reward_source="caregiver",
    configure=configure_babbling,
)

GUIDED_BABBLING_STAGE = TrainingStage(
    name="guided_babbling",
    description="Continued motor learning with caregiver reward + T2 context",
    n_tokens=500_000,
    learning_regions=["M1"],
    load_checkpoint="stage2_babbling",
    save_checkpoint="stage3_guided",
    exploration_noise=0.5,
    force_motor_active=True,
    reward_source="caregiver",
    configure=configure_guided_babbling,
)
