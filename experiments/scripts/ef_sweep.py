"""Quick sweep of efference copy gain values in REPL-like generation."""

import sys

sys.path.insert(0, "src")

from step.config import CortexConfig, make_motor_region, make_sensory_region
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import RewardModulator, SurpriseTracker, ThalamicGate
from step.cortex.topology import Topology
from step.data import EOM_TOKEN, prepare_tokens_personachat
from step.encoders.positional import PositionalCharEncoder


def build_model(alphabet):
    encoder = PositionalCharEncoder(alphabet)
    cfg_s1 = CortexConfig(
        n_columns=128,
        n_l4=4,
        n_l23=4,
        k_columns=8,
        ltd_rate=0.05,
        synapse_decay=1.0,
        learning_rate=0.05,
    )
    cfg_m1 = CortexConfig(
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
        ltd_rate=0.05,
        synapse_decay=1.0,
        learning_rate=0.05,
    )
    cfg_s2 = CortexConfig(
        n_columns=32,
        n_l4=4,
        n_l23=4,
        k_columns=4,
        ltd_rate=0.05,
        synapse_decay=1.0,
        learning_rate=0.05,
    )

    s1 = make_sensory_region(
        cfg_s1,
        encoder.input_dim,
        encoding_width=encoder.encoding_width,
    )
    s2 = make_sensory_region(cfg_s2, s1.n_l23_total * 4, seed=1)  # *4 for buffer_depth
    m1 = make_motor_region(cfg_m1, s1.n_l23_total, seed=2)

    cortex = Topology(encoder)
    cortex.add_region("S1", s1, entry=True)
    cortex.add_region("S2", s2)
    cortex.add_region("M1", m1, basal_ganglia=BasalGanglia(cfg_s1.n_columns + 1))

    cortex.connect(
        "S1",
        "S2",
        "feedforward",
        buffer_depth=4,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S1", "M1", "feedforward")
    cortex.connect(
        "M1",
        "S1",
        "apical",
        thalamic_gate=ThalamicGate(),
        reward_modulator=RewardModulator(),
    )

    return cortex, encoder, s1, m1


def step_token(cortex, token_id, token_str):
    cortex.step(token_id, token_str)


def generate(cortex, s1, m1, encoder, prompt, max_steps=30):
    """Feed prompt, inject EOM, then autoregress with forced gate."""
    # Input phase
    last_token = (ord(" "), " ")
    for ch in prompt:
        step_token(cortex, ord(ch), ch)
        last_token = (ord(ch), ch)

    # EOM injection
    step_token(cortex, EOM_TOKEN, "")

    # Generation phase
    cortex.force_gate_open = True
    spoken = []
    silent = 0

    for _ in range(max_steps + 10):
        step_token(cortex, last_token[0], last_token[1])
        m_id, _m_conf = m1.last_output

        if m_id >= 0:
            ch = chr(m_id) if 32 <= m_id < 127 else "?"
            spoken.append(ch)
            last_token = (m_id, ch)
            silent = 0
            if len(spoken) >= max_steps:
                break
        else:
            silent += 1
            if spoken and silent >= 3:
                break
            if silent >= 10:
                break

    cortex.force_gate_open = False
    return "".join(spoken)


def main():
    print("Loading PersonaChat vocabulary...")
    tokens = prepare_tokens_personachat(100000, speak_window=5)
    alphabet = sorted({ch for _, ch in tokens if _ >= 0})

    gains = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    prompts = ["hello how are you", "i like to play guitar", "what do you do"]

    for gain in gains:
        print(f"\n{'=' * 60}")
        print(f"  efference_gain = {gain}")
        print(f"{'=' * 60}")

        # Build fresh model + load checkpoint for each gain
        cortex, encoder, s1, m1 = build_model(alphabet)
        cortex.load_checkpoint("experiments/checkpoints/personachat_k4_100k.ckpt")

        # Set the gain
        s1.efference_gain = gain

        for prompt in prompts:
            # Reset working memory between prompts
            for name in ["S1", "S2", "M1"]:
                cortex._regions[name].region.reset_working_memory()
            encoder.reset()

            output = generate(cortex, s1, m1, encoder, prompt, max_steps=40)
            unique = len(set(output))
            print(f"  '{prompt}' → '{output}'  ({len(output)} chars, {unique} unique)")


if __name__ == "__main__":
    main()
