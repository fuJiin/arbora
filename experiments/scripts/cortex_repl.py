#!/usr/bin/env python3
"""Interactive REPL for exploring cortex model behavior.

Trains a model on TinyDialogues, then enters interactive mode where
you type text and see the model's predictions, surprise, and internal
state in real time. The model continues learning from your input.

Usage:
    uv run experiments/scripts/cortex_repl.py
    uv run experiments/scripts/cortex_repl.py --warmup 10000
    uv run experiments/scripts/cortex_repl.py --no-warmup
"""

import argparse
import math
import sys

import numpy as np

import step.env  # noqa: F401
from step.config import (
    _default_motor_config,
    _default_region2_config,
    _default_s1_config,
    make_motor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.topology import Topology
from step.data import prepare_tokens_tinydialogues
from step.decoders.dendritic import DendriticDecoder  # for type hints
from step.encoders.positional import PositionalCharEncoder

# ANSI colors
DIM = "\033[2m"
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"


def surprise_color(bits: float, vocab_size: int = 65) -> str:
    """Color-code by surprise level."""
    random_bpc = math.log2(max(vocab_size, 2))
    if bits < random_bpc * 0.5:
        return GREEN  # Well predicted
    if bits < random_bpc * 0.75:
        return YELLOW  # Somewhat surprising
    return RED  # Very surprising


def build_model(alphabet: str):
    """Build full hierarchy with tuned S1 config."""
    encoder = PositionalCharEncoder(alphabet, max_positions=8)

    s1_cfg = _default_s1_config()
    region1 = make_sensory_region(
        s1_cfg, encoder.input_dim, encoder.encoding_width,
    )

    r2_cfg = _default_region2_config()
    r2_input_dim = region1.n_l23_total * 4
    region2 = make_sensory_region(r2_cfg, r2_input_dim, seed=123)

    m1_cfg = _default_motor_config()
    motor = make_motor_region(m1_cfg, region1.n_l23_total, seed=456)

    bg = BasalGanglia(
        context_dim=region1.n_columns + 1,
        learning_rate=0.1,
        seed=789,
    )

    cortex = Topology(encoder, diagnostics_interval=999999)
    cortex.add_region("S1", region1, entry=True)
    cortex.add_region("S2", region2)
    cortex.connect(
        "S1", "S2", "feedforward", buffer_depth=4, burst_gate=True,
    )
    cortex.connect("S1", "S2", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.add_region("M1", motor, basal_ganglia=bg)
    cortex.connect("S1", "M1", "feedforward")
    cortex.connect("S1", "M1", "surprise", surprise_tracker=SurpriseTracker())
    if motor.n_l23_total == region2.n_l23_total:
        cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())

    # The topology creates its own DendriticDecoder for the entry region
    # during add_region(). We'll grab the reference after construction.
    entry_state = cortex._regions["S1"]
    decoder = entry_state.dendritic_decoder

    return cortex, encoder, region1, motor, decoder


def warmup(cortex, tokens, log_interval=2000):
    """Train the model on TinyDialogues tokens."""
    n = len(tokens)
    print(f"{DIM}Warming up on {n:,} chars...{RESET}")
    import contextlib
    import io

    # Suppress verbose representation reports during warmup
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = cortex.run(tokens, log_interval=log_interval)
    s1 = result.per_region["S1"]
    print(
        f"{DIM}Warmup complete: "
        f"BPC={s1.bpc:.2f} "
        f"recent={s1.bpc_recent:.2f} "
        f"burst={s1.representation.get('burst_rate', 0):.0%}{RESET}\n"
    )
    return result


def decode_prediction(
    l23_state: np.ndarray,
    decoder: DendriticDecoder,
    encoder: PositionalCharEncoder,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Get top-k predicted characters with confidence scores."""
    scores = decoder.decode_scores(l23_state)
    if not scores:
        return []

    # Softmax over scores
    raw = np.array(list(scores.values()), dtype=np.float64)
    keys = list(scores.keys())
    raw -= raw.max()
    exp_scores = np.exp(raw)
    total = exp_scores.sum()

    predictions = []
    for i in np.argsort(exp_scores)[::-1][:k]:
        token_id = keys[i]
        prob = exp_scores[i] / total
        # Map token_id (ord) back to char
        ch = chr(token_id) if 32 <= token_id < 127 else f"<{token_id}>"
        predictions.append((ch, prob))

    return predictions


def compute_bits(
    token_id: int,
    l23_state: np.ndarray,
    decoder: DendriticDecoder,
) -> float:
    """Compute bits (surprise) for one character."""
    scores = decoder.decode_scores(l23_state)
    n_tokens = decoder.n_tokens

    if n_tokens == 0 or not scores:
        return math.log2(max(n_tokens, 2)) if n_tokens > 0 else 6.0

    raw = np.array(list(scores.values()), dtype=np.float64)
    keys = list(scores.keys())
    raw -= raw.max()
    exp_scores = np.exp(raw)
    n_unseen = max(n_tokens - len(scores), 0)
    floor = 0.01 * exp_scores.mean() if n_unseen > 0 else 0.0
    total = exp_scores.sum() + n_unseen * floor

    if token_id in scores:
        idx = keys.index(token_id)
        prob = exp_scores[idx] / total
    else:
        prob = floor / total if floor > 0 else 1.0 / max(n_tokens, 2)

    return -math.log2(max(prob, 1e-10))


def format_predictions(preds: list[tuple[str, float]]) -> str:
    """Format prediction list for display."""
    if not preds:
        return f"{DIM}(no predictions){RESET}"
    parts = []
    for ch, prob in preds[:5]:
        display = repr(ch) if ch in ("\n", "\t", " ") else ch
        parts.append(f"{display}:{prob:.0%}")
    return " ".join(parts)


def interactive_loop(cortex, encoder, region1, motor, decoder):
    """Main REPL loop: type text, see predictions + surprise."""
    print(f"{BOLD}=== STEP Cortex REPL ==={RESET}")
    print(f"{DIM}Type text to see model predictions and surprise.")
    print(f"Commands: /reset (clear memory), /stats, /quit{RESET}\n")

    total_bits = 0.0
    n_chars = 0
    recent_bits = []

    while True:
        try:
            line = input(f"{CYAN}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        # Commands
        if line.startswith("/"):
            cmd = line.strip().lower()
            if cmd == "/quit" or cmd == "/q":
                break
            elif cmd == "/reset":
                for s in cortex._regions.values():
                    s.region.reset_working_memory()
                if hasattr(encoder, "reset"):
                    encoder.reset()
                total_bits = 0.0
                n_chars = 0
                recent_bits.clear()
                print(f"{DIM}Working memory cleared.{RESET}")
                continue
            elif cmd == "/stats":
                avg_bpc = total_bits / n_chars if n_chars > 0 else 0
                recent_bpc = (
                    sum(recent_bits) / len(recent_bits)
                    if recent_bits else 0
                )
                print(
                    f"{DIM}Chars: {n_chars}  "
                    f"BPC: {avg_bpc:.2f}  "
                    f"Recent: {recent_bpc:.2f}  "
                    f"Decoder tokens: {decoder.n_tokens}{RESET}"
                )
                continue
            else:
                print(f"{DIM}Unknown command: {cmd}{RESET}")
                continue

        # Process each character
        print()
        for ch in line:
            token_id = ord(ch)

            # Get prediction BEFORE processing (using previous L2/3 state)
            preds = decode_prediction(
                region1.active_l23, decoder, encoder,
            )
            bits = compute_bits(token_id, region1.active_l23, decoder)

            # Update decoder with current state -> token mapping
            decoder.observe(token_id, region1.active_l23)

            # Process the character through the full hierarchy
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                cortex.run([(token_id, ch)], log_interval=0)

            # Accumulate stats
            total_bits += bits
            n_chars += 1
            recent_bits.append(bits)
            if len(recent_bits) > 100:
                recent_bits.pop(0)

            # Display
            color = surprise_color(bits)
            pred_str = format_predictions(preds)

            # Show character with surprise coloring
            display_ch = repr(ch) if ch == " " else ch
            sys.stdout.write(
                f"  {color}{display_ch}{RESET} "
                f"{DIM}{bits:.1f}b{RESET}  "
                f"{DIM}pred: {pred_str}{RESET}\n"
            )

        # Show summary for the line
        if n_chars > 0:
            recent_bpc = (
                sum(recent_bits) / len(recent_bits)
                if recent_bits else 0
            )
            print(
                f"\n{DIM}  BPC: {total_bits / n_chars:.2f} "
                f"(recent: {recent_bpc:.2f}){RESET}\n"
            )

        # Insert story boundary between lines to reset context
        # (simulates new dialogue)
        for s in cortex._regions.values():
            s.region.reset_working_memory()
        if hasattr(encoder, "reset"):
            encoder.reset()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive cortex REPL",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5000,
        help="Warmup chars from TinyDialogues (0 or --no-warmup to skip)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup, start with fresh model",
    )
    args = parser.parse_args()

    # Build alphabet from TinyDialogues (need consistent vocab)
    print(f"{DIM}Loading vocabulary...{RESET}")
    sample = prepare_tokens_tinydialogues(1000, speak_window=5)
    alphabet = sorted({ch for _, ch in sample if _ >= 0})
    alphabet_str = "".join(alphabet)
    print(f"{DIM}Vocab: {len(alphabet)} chars{RESET}")

    # Build model
    print(f"{DIM}Building model (S1=128/k=8, S2, M1+BG)...{RESET}")
    cortex, encoder, region1, motor, decoder = build_model(alphabet_str)

    # Warmup
    if not args.no_warmup and args.warmup > 0:
        tokens = prepare_tokens_tinydialogues(
            args.warmup, speak_window=10,
        )
        warmup(cortex, tokens)

    # Enter interactive mode
    interactive_loop(cortex, encoder, region1, motor, decoder)
    print(f"{DIM}Goodbye!{RESET}")


if __name__ == "__main__":
    main()
