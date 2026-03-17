"""Training stage configuration for developmental learning.

Each stage defines which regions learn, which connections are active,
and what reward signals are used. Stages are applied to a Topology
via configure(), and checkpoints flow between stages.

Stages model infant development:
  1. Sensory:  S1→S2→S3 representation learning
  2. Babbling: M1→S1→M1 motor exploration (S1 frozen)
  3. Guided:   M1+BG+S2 word-level RL
  4. Imitation: S1→S2→M2→M1 echolalia
  5. Generation: PFC→M2→M1 goal-directed RL
"""

from dataclasses import dataclass, field


@dataclass
class StageConnection:
    """Declares whether a connection should be active in this stage."""

    source: str
    target: str
    kind: str
    enabled: bool = True


@dataclass
class TrainingStage:
    """Configuration for a training stage."""

    name: str
    description: str

    # Tokens to train for this stage
    n_tokens: int = 100_000

    # Regions where learning is enabled (all others frozen)
    learning_regions: list[str] = field(default_factory=list)

    # Connection overrides (only listed connections are changed;
    # unlisted connections keep their current state)
    connections: list[StageConnection] = field(default_factory=list)

    # Checkpoint to load before this stage (None = start fresh or continue)
    load_checkpoint: str | None = None

    # Checkpoint to save after this stage
    save_checkpoint: str | None = None

    # Motor babbling noise (0.0 = off, 1.0 = pure random)
    babbling_noise: float = 0.0

    # Force M1 active every step (not just EOM phase)
    force_motor_active: bool = False

    # Reward source: "turn_taking" (default) or "word" (S2 recognition)
    reward_source: str = "turn_taking"

    def configure(self, topology) -> None:
        """Apply this stage's configuration to a Topology."""
        # Freeze/unfreeze regions
        for name in topology._regions:
            if self.learning_regions:
                if name in self.learning_regions:
                    topology.unfreeze_region(name)
                else:
                    topology.freeze_region(name)

        # Motor babbling configuration
        for name, state in topology._regions.items():
            if state.motor:
                state.region.babbling_noise = self.babbling_noise
        topology.force_gate_open = self.force_motor_active

        # Reward source configuration
        if self.reward_source == "curiosity":
            from step.cortex.reward import CuriosityReward
            topology.set_reward_source(CuriosityReward())
        elif self.reward_source == "word":
            s2_cols = 32
            for name, state in topology._regions.items():
                if name == "S2":
                    s2_cols = state.region.n_columns
                    break
            from step.cortex.reward import WordReward
            topology.set_reward_source(WordReward(s2_cols))
        else:
            topology.set_reward_source(None)  # default turn-taking

        # Enable/disable connections
        for sc in self.connections:
            try:
                if sc.enabled:
                    topology.enable_connection(sc.source, sc.target, sc.kind)
                else:
                    topology.disable_connection(sc.source, sc.target, sc.kind)
            except ValueError:
                pass  # Connection doesn't exist in this topology — skip


# ------------------------------------------------------------------
# Predefined stages
# ------------------------------------------------------------------

def _sensory_connections():
    """Sensory stage: S1→S2→S3 active, M1 disconnected."""
    return [
        # Sensory feedforward + surprise: on
        StageConnection("S1", "S2", "feedforward", enabled=True),
        StageConnection("S1", "S2", "surprise", enabled=True),
        StageConnection("S2", "S3", "feedforward", enabled=True),
        StageConnection("S2", "S3", "surprise", enabled=True),
        # Apical feedback: on
        StageConnection("S2", "S1", "apical", enabled=True),
        StageConnection("S3", "S2", "apical", enabled=True),
        # M1: off
        StageConnection("S1", "M1", "feedforward", enabled=False),
        StageConnection("S1", "M1", "surprise", enabled=False),
        StageConnection("M1", "S1", "apical", enabled=False),
    ]


def _babbling_connections():
    """Babbling stage: S1→M1 active, everything else off."""
    return [
        # S1→M1: on
        StageConnection("S1", "M1", "feedforward", enabled=True),
        StageConnection("S1", "M1", "surprise", enabled=True),
        # M1→S1 apical: off (M1 not useful yet)
        StageConnection("M1", "S1", "apical", enabled=False),
        # Higher regions: off
        StageConnection("S1", "S2", "feedforward", enabled=False),
        StageConnection("S1", "S2", "surprise", enabled=False),
        StageConnection("S2", "S3", "feedforward", enabled=False),
        StageConnection("S2", "S3", "surprise", enabled=False),
        StageConnection("S2", "S1", "apical", enabled=False),
        StageConnection("S3", "S2", "apical", enabled=False),
    ]


def _guided_connections():
    """Guided babbling: S1→M1 + S1→S2 (for word reward), apical on."""
    return [
        # S1→M1: on
        StageConnection("S1", "M1", "feedforward", enabled=True),
        StageConnection("S1", "M1", "surprise", enabled=True),
        # S1→S2: on (S2 provides reward signal)
        StageConnection("S1", "S2", "feedforward", enabled=True),
        StageConnection("S1", "S2", "surprise", enabled=True),
        # S2→S1 apical: on (word context helps)
        StageConnection("S2", "S1", "apical", enabled=True),
        # S3: off
        StageConnection("S2", "S3", "feedforward", enabled=False),
        StageConnection("S2", "S3", "surprise", enabled=False),
        StageConnection("S3", "S2", "apical", enabled=False),
        # M1→S1 apical: off
        StageConnection("M1", "S1", "apical", enabled=False),
    ]


SENSORY_STAGE = TrainingStage(
    name="sensory",
    description="Self-supervised sensory representation learning (S1→S2→S3)",
    n_tokens=1_000_000,
    learning_regions=["S1", "S2", "S3"],
    connections=_sensory_connections(),
    save_checkpoint="stage1_sensory",
)

BABBLING_STAGE = TrainingStage(
    name="babbling",
    description="Motor babbling with curiosity reward (M1→S1→M1, S1 frozen)",
    n_tokens=200_000,
    learning_regions=["M1"],
    connections=_babbling_connections(),
    load_checkpoint="stage1_sensory",
    save_checkpoint="stage2_babbling",
    babbling_noise=0.5,  # Start with exploration, adaptive from there
    force_motor_active=True,
    reward_source="curiosity",  # RPE drives learning from step 1
)

GUIDED_BABBLING_STAGE = TrainingStage(
    name="guided_babbling",
    description="Continued motor learning with curiosity + S2 context",
    n_tokens=500_000,
    learning_regions=["M1"],
    connections=_guided_connections(),
    load_checkpoint="stage2_babbling",
    save_checkpoint="stage3_guided",
    babbling_noise=0.5,
    force_motor_active=True,
    reward_source="curiosity",
)
