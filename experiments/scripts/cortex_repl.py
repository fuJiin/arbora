#!/usr/bin/env python3
"""Interactive REPL for exploring cortex model behavior.

Trains a model on TinyDialogues, then enters interactive mode where
you type text and see the model's predictions, surprise, and internal
state in real time. Uses the full S1→S2→M1+BG architecture:

- Input phase: each char processed through hierarchy, M1 interruptions shown
- EOM injection: after your input, M1 gets a chance to speak
- Speaking phase: M1 output fed back as input until EOM or ramble limit

Usage:
    uv run experiments/scripts/cortex_repl.py
    uv run experiments/scripts/cortex_repl.py --warmup 10000
    uv run experiments/scripts/cortex_repl.py --no-warmup
"""

import argparse
import contextlib
import io
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
from step.data import (
    EOM_TOKEN,
    prepare_tokens_personachat,
    prepare_tokens_tinydialogues,
)
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

# Thresholds for speaking phase
MAX_SPEAK_STEPS = 200   # max chars M1 can generate per turn
MAX_SILENT_STEPS = 10   # give up waiting for M1 after this many silent steps


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

    entry_state = cortex._regions["S1"]
    decoder = entry_state.dendritic_decoder

    return cortex, encoder, region1, motor, decoder


def warmup(cortex, tokens, log_interval=2000):
    """Train the model on TinyDialogues tokens."""
    n = len(tokens)
    print(f"{DIM}Warming up on {n:,} chars...{RESET}")

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


def step_token(cortex, token_id, token_str):
    """Process one token through the full hierarchy, return motor output."""
    cortex.step(token_id, token_str)


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

    raw = np.array(list(scores.values()), dtype=np.float64)
    keys = list(scores.keys())
    raw -= raw.max()
    exp_scores = np.exp(raw)
    total = exp_scores.sum()

    predictions = []
    for i in np.argsort(exp_scores)[::-1][:k]:
        token_id = keys[i]
        prob = exp_scores[i] / total
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


def token_to_char(token_id: int) -> str | None:
    """Convert token_id back to printable char, or None."""
    if 32 <= token_id < 127:
        return chr(token_id)
    return None


CHECKPOINT_DIR = "experiments/checkpoints"


def print_help():
    """Print available commands."""
    print(f"{DIM}Commands:{RESET}")
    print(f"{DIM}  /help            Show this help{RESET}")
    print(f"{DIM}  /reset           Clear working memory{RESET}")
    print(f"{DIM}  /stats           Show BPC statistics{RESET}")
    print(
        f"{DIM}  /warmup [N]      "
        f"Train on N more TinyDialogues chars (default 5000){RESET}"
    )
    print(
        f"{DIM}  /save [name]     "
        f"Save checkpoint (default: repl_default){RESET}"
    )
    print(
        f"{DIM}  /load [name]     "
        f"Load checkpoint (default: repl_default){RESET}"
    )
    print(f"{DIM}  /quit, /q        Exit{RESET}")
    print()
    print(f"{DIM}Type text to feed it through S1→S2→M1+BG.{RESET}")
    print(
        f"{DIM}After your input, EOM is injected "
        f"and M1 gets a turn to speak.{RESET}"
    )
    print(
        f"{DIM}M1 interruptions during input "
        f"are shown inline.{RESET}"
    )


def reset_state(cortex, encoder):
    """Reset working memory across all regions (story boundary)."""
    for s in cortex._regions.values():
        s.region.reset_working_memory()
        if s.basal_ganglia is not None:
            s.basal_ganglia.reset()
    for conn in cortex._connections:
        if conn._buffer is not None:
            conn._buffer[:] = 0.0
            conn._buffer_pos = 0
        if conn.thalamic_gate is not None:
            conn.thalamic_gate.reset()
    cortex._in_eom = False
    cortex._eom_steps = 0
    if hasattr(encoder, "reset"):
        encoder.reset()


def interactive_loop(cortex, encoder, region1, motor, decoder, load_fn):
    """Main REPL loop with full M1+BG turn-taking."""
    print(f"{BOLD}=== STEP Cortex REPL ==={RESET}")
    print_help()
    print()

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
            parts = line.strip().split()
            cmd = parts[0].lower()
            if cmd in ("/quit", "/q"):
                break
            elif cmd == "/help":
                print_help()
                continue
            elif cmd == "/reset":
                reset_state(cortex, encoder)
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
            elif cmd == "/warmup":
                n = int(parts[1]) if len(parts) > 1 else 5000
                tokens = load_fn(n, speak_window=10)
                warmup(cortex, tokens)
                continue
            elif cmd == "/save":
                name = parts[1] if len(parts) > 1 else "repl_default"
                import os
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                path = os.path.join(CHECKPOINT_DIR, f"{name}.ckpt")
                cortex.save_checkpoint(path)
                print(f"{DIM}Saved checkpoint: {path}{RESET}")
                continue
            elif cmd == "/load":
                name = parts[1] if len(parts) > 1 else "repl_default"
                import os
                path = os.path.join(CHECKPOINT_DIR, f"{name}.ckpt")
                if not os.path.exists(path):
                    print(f"{RED}Checkpoint not found: {path}{RESET}")
                else:
                    cortex.load_checkpoint(path)
                    print(f"{DIM}Loaded checkpoint: {path}{RESET}")
                continue
            else:
                print(f"{DIM}Unknown command: {cmd} (try /help){RESET}")
                continue

        # ── INPUT PHASE ──
        # Process each character through full hierarchy.
        # M1 is active but BG should keep gate closed (input phase).
        # Any M1 output = interruption.
        print()
        interruptions = 0
        line_correct = 0
        line_total = 0
        line_bits = []
        last_token = (ord(" "), " ")

        for ch in line:
            token_id = ord(ch)
            last_token = (token_id, ch)

            # S1 prediction BEFORE processing
            preds = decode_prediction(
                region1.active_l23, decoder, encoder,
            )
            bits = compute_bits(token_id, region1.active_l23, decoder)

            # Train decoder on current state → token
            decoder.observe(token_id, region1.active_l23)

            # Step through full hierarchy
            step_token(cortex, token_id, ch)

            # Check M1 output (after BG gating)
            m_id, m_conf = motor.last_output
            gate = motor.last_gate

            # Track prediction accuracy (did top-1 match?)
            line_total += 1
            if preds and preds[0][0] == ch:
                line_correct += 1
            line_bits.append(bits)

            # Accumulate global stats
            total_bits += bits
            n_chars += 1
            recent_bits.append(bits)
            if len(recent_bits) > 100:
                recent_bits.pop(0)

            # Display input char with aligned columns
            color = surprise_color(bits)
            pred_str = format_predictions(preds)
            display_ch = repr(ch) if ch == " " else ch

            m1_info = ""
            if m_id >= 0:
                interruptions += 1
                m1_ch = token_to_char(m_id)
                m1_display = m1_ch if m1_ch else f"<{m_id}>"
                m1_info = (
                    f"  {RED}!! M1: '{m1_display}'{RESET}"
                )

            # Aligned columns: char | bits | gate | predictions
            sys.stdout.write(
                f"  {color}{display_ch:<4s}{RESET}"
                f" {DIM}{bits:5.1f} bits{RESET}"
                f"  {DIM}gate {gate:.2f}{RESET}"
                f"  {DIM}{pred_str}{RESET}"
                f"{m1_info}\n"
            )

        # ── EOM INJECTION ──
        # Signal turn boundary: M1's turn to speak.
        # Feed EOM token, then neutral input (repeat last char) as speak window.
        step_token(cortex, EOM_TOKEN, "")

        # Show transition with summary stats
        if n_chars > 0:
            recent_bpc = (
                sum(recent_bits) / len(recent_bits)
                if recent_bits else 0
            )
            line_bpc = (
                sum(line_bits) / len(line_bits)
                if line_bits else 0
            )
            line_acc = (
                line_correct / line_total
                if line_total > 0 else 0
            )
            burst_pct = (
                float(region1.bursting_columns.sum())
                / max(region1.n_columns, 1)
            )
            print(
                f"\n{DIM}  line: {line_bpc:.2f} bpc, "
                f"{line_acc:.0%} acc, "
                f"{burst_pct:.0%} burst  |  "
                f"overall: {total_bits / n_chars:.2f} bpc "
                f"(recent: {recent_bpc:.2f})  "
                f"interruptions: {interruptions}{RESET}"
            )
        print(f"\n{DIM}  [EOM → M1's turn]{RESET}")

        # ── SPEAKING PHASE ──
        # Force BG gate open — we know it's M1's turn. BG gating
        # is for learned turn-taking; in the REPL we control turns.
        # M1 still processes normally, we just bypass the gate filter.
        cortex.force_gate_open = True
        spoken_chars = []
        silent_steps = 0

        for _ in range(MAX_SPEAK_STEPS + MAX_SILENT_STEPS):
            # Feed neutral input (last char repeated, like TinyDialogues pattern)
            step_token(cortex, last_token[0], last_token[1])

            m_id, m_conf = motor.last_output
            gate = motor.last_gate

            if m_id >= 0:
                # M1 is speaking
                ch = token_to_char(m_id)
                if ch:
                    spoken_chars.append(ch)
                    color = GREEN if m_conf > 0.5 else YELLOW
                    sys.stdout.write(f"{color}{ch}{RESET}")
                    sys.stdout.flush()
                    # Feed M1's output as the next input (autoregressive)
                    last_token = (m_id, ch)
                silent_steps = 0

                # Ramble check
                if len(spoken_chars) >= MAX_SPEAK_STEPS:
                    sys.stdout.write(
                        f"\n{DIM}  [ramble limit: {MAX_SPEAK_STEPS} chars]{RESET}"
                    )
                    break
            else:
                silent_steps += 1
                if spoken_chars:
                    # Was speaking, now stopped → natural end
                    break
                if silent_steps >= MAX_SILENT_STEPS:
                    sys.stdout.write(
                        f"{DIM}  (M1 silent — gate={gate:.2f}){RESET}"
                    )
                    break

        cortex.force_gate_open = False

        if spoken_chars:
            sys.stdout.write(
                f"\n{DIM}  M1 spoke {len(spoken_chars)} chars{RESET}"
            )
        print("\n")

        # Reset working memory for next exchange
        reset_state(cortex, encoder)


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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load checkpoint (name or path) instead of warmup",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="personachat",
        choices=["personachat", "tinydialogues"],
        help="Dataset for vocab and warmup (default: personachat)",
    )
    args = parser.parse_args()

    # Load tokens for vocab (and warmup)
    load_fn = (
        prepare_tokens_personachat
        if args.dataset == "personachat"
        else prepare_tokens_tinydialogues
    )

    # Need enough chars to discover full alphabet (rare chars like digits
    # only appear after many dialogues). 100k when loading checkpoint,
    # smaller sample otherwise since warmup will cover it.
    vocab_size = 100000 if args.checkpoint else 10000
    print(f"{DIM}Loading vocabulary from {args.dataset}...{RESET}")
    sample = load_fn(vocab_size, speak_window=5)
    alphabet = sorted({ch for _, ch in sample if _ >= 0})
    alphabet_str = "".join(alphabet)
    print(f"{DIM}Vocab: {len(alphabet)} chars{RESET}")

    # Build model
    print(f"{DIM}Building model (S1=128/k=8, S2, M1+BG)...{RESET}")
    cortex, encoder, region1, motor, decoder = build_model(alphabet_str)

    # Load checkpoint or warmup
    if args.checkpoint:
        import os
        path = args.checkpoint
        if not os.path.exists(path):
            path = os.path.join(CHECKPOINT_DIR, f"{args.checkpoint}.ckpt")
        if os.path.exists(path):
            cortex.load_checkpoint(path)
            print(f"{DIM}Loaded checkpoint: {path}{RESET}\n")
        else:
            print(f"{RED}Checkpoint not found: {path}{RESET}")
            sys.exit(1)
    elif not args.no_warmup and args.warmup > 0:
        tokens = load_fn(args.warmup, speak_window=10)
        warmup(cortex, tokens)

    # Enter interactive mode
    interactive_loop(cortex, encoder, region1, motor, decoder, load_fn)
    print(f"{DIM}Goodbye!{RESET}")


if __name__ == "__main__":
    main()
