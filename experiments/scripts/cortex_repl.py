#!/usr/bin/env python3
"""Interactive REPL for exploring cortex model behavior.

Trains a model, then enters interactive mode where you type text and
see the model's predictions, surprise, and internal state in real time.
Uses the full S1→S2→S3→PFC→M2→M1+BG architecture:

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
from step.agent import ChatAgent
from step.data import (
    EOM_TOKEN,
    prepare_tokens_personachat,
    prepare_tokens_tinydialogues,
)
from step.decoders.dendritic import DendriticDecoder  # for type hints
from step.encoders.positional import PositionalCharEncoder
from step.environment import EOM_OBS, ChatEnv, ChatObs
from step.train import train

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
MAX_SPEAK_STEPS = 200  # max chars M1 can generate per turn
MAX_SILENT_STEPS = 10  # give up waiting for M1 after this many silent steps


def surprise_color(burst_frac: float) -> str:
    """Color-code by burst fraction (surprise)."""
    if burst_frac < 0.3:
        return GREEN  # Well predicted
    if burst_frac < 0.6:
        return YELLOW  # Somewhat surprising
    return RED  # Very surprising


def build_model(alphabet: str, *, use_l5_apical: bool = False):
    """Build full hierarchy using the canonical circuit factory.

    Uses build_canonical_circuit() to ensure dimensions always match
    checkpoints saved by the staged pipeline.
    """
    from step.cortex.canonical import build_canonical_circuit

    encoder = PositionalCharEncoder(alphabet, max_positions=8)

    apical_override = {"use_l5_apical_segments": True} if use_l5_apical else None
    cortex = build_canonical_circuit(
        encoder,
        log_interval=999999,
        timeline_interval=0,
        s1_overrides=apical_override,
        s2_overrides=apical_override,
        s3_overrides=apical_override,
    )

    region1 = cortex._regions["S1"].region
    motor = cortex._regions["M1"].region
    decoder = cortex._regions["S1"].dendritic_decoder
    word_decoder = cortex._regions["S2"].word_decoder
    agent = ChatAgent(encoder=encoder, circuit=cortex)

    return cortex, encoder, region1, motor, decoder, word_decoder, agent


def warmup(cortex, encoder, tokens, log_interval=2000):
    """Train the model on TinyDialogues tokens."""
    n = len(tokens)
    print(f"{DIM}Warming up on {n:,} chars...{RESET}")

    env = ChatEnv(tokens)
    agent = ChatAgent(encoder=encoder, circuit=cortex)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = train(env, agent, log_interval=log_interval)
    s1 = result.per_region["S1"]
    print(
        f"{DIM}Warmup complete: "
        f"BPC={s1.bpc:.2f} "
        f"recent={s1.bpc_recent:.2f} "
        f"burst={s1.representation.get('burst_rate', 0):.0%}{RESET}\n"
    )
    return result


def step_token(agent, token_id, token_str):
    """Process one token through the full hierarchy via ChatAgent."""
    if token_id == EOM_TOKEN:
        agent.act(EOM_OBS, 0.0)
    else:
        obs = ChatObs(token_id=token_id, token_str=token_str)
        agent.act(obs, 0.0)


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
    print(f"{DIM}  /info            Show model capabilities and sample prompts{RESET}")
    print(f"{DIM}  /reset           Clear working memory{RESET}")
    print(f"{DIM}  /stats           Show BPC statistics{RESET}")
    print(f"{DIM}  /warmup [N]      Train on N more chars (default 5000){RESET}")
    print(f"{DIM}  /save [name]     Save checkpoint (default: repl_default){RESET}")
    print(f"{DIM}  /load [name]     Load checkpoint (default: repl_default){RESET}")
    print(f"{DIM}  /babble [N]     Watch M1 babble N chars (default 200){RESET}")
    print(f"{DIM}  /probe          Show all region representation quality{RESET}")
    print(f"{DIM}  /echo [word]    Hear a word, watch M1 try to reproduce it{RESET}")
    print(f"{DIM}  /quit, /q        Exit{RESET}")
    print()
    print(f"{DIM}Type text to feed through S1→S2→S3→PFC→M2→M1.{RESET}")
    print(f"{DIM}After your input, EOM is injected and M1 gets a turn to speak.{RESET}")
    print(f"{DIM}M1 interruptions during input are shown inline.{RESET}")


def print_info(cortex, encoder, region1, motor, decoder):
    """Show model capabilities, vocabulary, and sample prompts."""
    print(f"\n{BOLD}=== Model Info ==={RESET}\n")

    # Architecture
    regions = list(cortex._regions.keys())
    frozen = [n for n, s in cortex._regions.items() if not s.region.learning_enabled]
    active_conns = [
        f"{c.source}→{c.target}({c.role.value})"
        for c in cortex._connections
        if c.enabled
    ]
    print(f"  {DIM}Regions:{RESET} {', '.join(regions)}")
    if frozen:
        print(f"  {DIM}Frozen:{RESET} {', '.join(frozen)}")
    print(f"  {DIM}Active connections:{RESET} {', '.join(active_conns) or 'none'}")

    # S1 stats
    burst_sum = float(region1.bursting_columns.sum())
    burst_pct = burst_sum / max(region1.n_columns, 1)
    print(f"  {DIM}S1 burst rate:{RESET} {burst_pct:.0%} (lower = better predictions)")

    # Vocabulary from decoder
    if decoder and decoder.n_tokens > 0:
        known = sorted(decoder._neurons.keys())
        chars = [chr(t) if 32 <= t < 127 else f"<{t}>" for t in known]
        print(f"  {DIM}Known chars ({len(chars)}):{RESET} {''.join(chars)}")

    # PFC status
    pfc_state = cortex._regions.get("PFC")
    if pfc_state is not None:
        pfc_r = pfc_state.region
        print(
            f"  {DIM}PFC:{RESET} {pfc_r.n_columns} cols, "
            f"k={pfc_r.k_columns}, "
            f"gate={'open' if pfc_r.gate_open else 'closed'}, "
            f"confidence={pfc_r.confidence:.3f}"
        )

    # M1 status
    if motor:
        l5_max = float(motor.output_weights.max(axis=0).max())
        n_active = int((motor.output_weights.max(axis=0) > 0.01).sum())
        babbling = getattr(motor, "babbling_noise", 0.0)
        print(
            f"  {DIM}M1 output_weights:{RESET} "
            f"{n_active}/{motor.n_output_tokens} active tokens, max={l5_max:.3f}"
        )
        if babbling > 0:
            print(f"  {DIM}M1 babbling noise:{RESET} {babbling:.0%}")

    # Sample prompts (BabyLM = child-directed speech)
    print(f"\n  {BOLD}Sample prompts (child-directed speech):{RESET}")
    samples = [
        "want milk?",
        "look at the dog!",
        "where is mommy?",
        "good boy!",
        "come here.",
        "what is that?",
        "no no no!",
        "all gone!",
        "hi baby!",
        "up up up!",
    ]
    # Pick 5 random samples
    rng = np.random.default_rng()
    for s in rng.choice(samples, min(5, len(samples)), replace=False):
        print(f"    {CYAN}> {s}{RESET}")

    print(f"\n  {DIM}Tip: Use short, simple phrases. This model was trained on{RESET}")
    print(f"  {DIM}child-directed speech — not adult conversation.{RESET}\n")


def reset_state(cortex):
    """Reset working memory across all regions (story boundary)."""
    cortex.reset()


def run_babble(agent, region1, motor, n_chars=200):
    """Watch M1 babble in real time using full circuit."""
    print(f"\n{DIM}  M1 babbling {n_chars} chars (PFC→M2→M1)...{RESET}")
    print(f"  {BOLD}", end="", flush=True)

    # Force gate open, set babbling noise for exploration
    old_gate = agent.force_gate_open
    old_noise = motor.babbling_noise
    agent.force_gate_open = True
    motor.babbling_noise = 0.3  # Mostly learned policy, some exploration

    current_char = " "
    produced = []
    word_buf = []

    for _i in range(n_chars):
        token_id = ord(current_char) if current_char else ord(" ")
        obs = ChatObs(token_id=token_id, token_str=current_char)
        agent.act(obs, 0.0)

        # M1 output
        pop_id, pop_conf = motor.get_population_output()
        if pop_id >= 0 and 32 <= pop_id < 127:
            ch = chr(pop_id)
            produced.append(ch)

            color = GREEN if pop_conf > 0.1 else YELLOW
            sys.stdout.write(f"{color}{ch}{RESET}{BOLD}")
            sys.stdout.flush()

            if ch in " .!?'-,":
                word_buf.clear()
            else:
                word_buf.append(ch)

            motor.observe_token(pop_id)
            current_char = ch
        else:
            sys.stdout.write(f"{DIM}_{RESET}{BOLD}")
            sys.stdout.flush()

    print(f"{RESET}")

    # Summary stats
    from collections import Counter

    chars = Counter(produced)
    n_unique = len(chars)
    bigrams = Counter(produced[i] + produced[i + 1] for i in range(len(produced) - 1))
    # Find real words
    text = "".join(produced)
    import re

    words = re.split(r"[ .!?',\-]+", text)
    words = [w for w in words if len(w) >= 2]

    burst_pct = float(region1.bursting_columns.sum()) / max(region1.n_columns, 1)
    print(
        f"\n{DIM}  {len(produced)} chars, {n_unique} unique, "
        f"burst={burst_pct:.0%}, "
        f"words: {len(words)} attempts{RESET}"
    )
    if bigrams:
        english_bg = ["th", "he", "in", "er", "an", "at", "is", "it", "to", "st", "ha"]
        found = [(bg, bigrams[bg]) for bg in english_bg if bg in bigrams]
        if found:
            bg_str = ", ".join(f"{bg}:{c}" for bg, c in found[:6])
            print(f"{DIM}  English bigrams: {bg_str}{RESET}")

    # Restore
    agent.force_gate_open = old_gate
    motor.babbling_noise = old_noise
    print()


def run_probe(cortex, word_decoder=None):
    """Show representation quality for all regions."""
    print(f"\n{BOLD}=== Region Probe ==={RESET}\n")
    for name, state in cortex._regions.items():
        r = state.region
        n_active = int(r.active_columns.sum())
        n_bursting = int(r.bursting_columns.sum())
        burst_pct = n_bursting / max(n_active, 1)

        # Weight stats
        ff_max = float(r.ff_weights.max())
        ff_sparsity = float((r.ff_weights == 0).mean())

        frozen = "frozen" if not r.learning_enabled else "learning"

        print(
            f"  {BOLD}{name}{RESET} ({frozen}): "
            f"burst={burst_pct:.0%} "
            f"ff_sparse={ff_sparsity:.1%} "
            f"ff_max={ff_max:.3f}"
        )

        # Motor-specific: L5 output weight stats
        if hasattr(r, "output_weights"):
            l5_max = float(r.output_weights.max())
            n_tokens = r.n_output_tokens
            # Which tokens have strongest weights?
            col_maxes = r.output_weights.max(axis=0)
            top_idx = col_maxes.argsort()[-5:][::-1]
            if hasattr(r, "_output_vocab") and r._output_vocab is not None:
                top_chars = [
                    chr(int(r._output_vocab[i]))
                    if 32 <= int(r._output_vocab[i]) < 127
                    else "?"
                    for i in top_idx
                ]
            else:
                top_chars = [chr(i) if 32 <= i < 127 else "?" for i in top_idx]
            print(
                f"    L5: {n_tokens} tokens, "
                f"max={l5_max:.3f}, "
                f"top chars: {' '.join(top_chars)}"
            )

        # PFC-specific: confidence and gate
        from step.cortex.pfc import PFCRegion

        if isinstance(r, PFCRegion):
            print(
                f"    PFC: gate={'open' if r.gate_open else 'closed'}, "
                f"confidence={r.confidence:.3f}, "
                f"trace_norm={float(np.abs(r._ff_eligibility).mean()):.5f}"
            )

        # Apical gain stats (per-source)
        if r._apical_sources:
            for src_name, src in r._apical_sources.items():
                w = src["weights"]
                print(
                    f"    Apical ({src_name}): "
                    f"mean={float(w.mean()):.4f} "
                    f"max={float(w.max()):.3f}"
                )

    # Word decoder stats
    if word_decoder and word_decoder.n_words > 0:
        print(f"  {BOLD}S2 word decoder{RESET}: {word_decoder.n_words} words")

    print()


def run_echo(agent, motor, word: str):
    """Interactive echo: hear a word, then watch M1 try to reproduce it.

    Demonstrates PFC goal maintenance and the full motor pipeline:
    1. Listen: process word through S1→S2→S3→PFC
    2. PFC snapshots goal, gate closes
    3. Speak: M1 produces chars, compare to target
    """
    from step.cortex.pfc import PFCRegion

    cortex = agent.circuit

    # Find PFC
    pfc = None
    for _name, s in cortex._regions.items():
        if isinstance(s.region, PFCRegion):
            pfc = s.region
            break

    print(f"\n{DIM}  Echo mode: listening to '{word}'...{RESET}")

    # Listen phase
    if pfc is not None:
        pfc.gate_open = True
    for ch in word:
        obs = ChatObs(token_id=ord(ch), token_str=ch)
        agent.act(obs, 0.0)

    # PFC snapshot
    if pfc is not None:
        pfc.snapshot_goal()
        pfc.gate_open = False
        print(f"{DIM}  PFC goal captured (confidence={pfc.confidence:.3f}){RESET}")

    # Speak phase
    print(f"  {DIM}M1 reproducing:{RESET} ", end="", flush=True)
    agent.force_gate_open = True
    old_noise = motor.babbling_noise
    motor.babbling_noise = 0.2  # Mostly policy, some exploration
    produced = []

    current = " "
    for i in range(len(word) + 2):
        # Use agent for processing (echo still uses low-level access
        # for per-char scoring — will be refactored into EchoEnv)
        obs = ChatObs(token_id=ord(current), token_str=current)
        agent.act(obs, 0.0)

        pop_id, _conf = motor.get_population_output()
        if pop_id >= 0 and 32 <= pop_id < 127:
            ch = chr(pop_id)
            produced.append(ch)
            # Color: green=exact match, yellow=in word, red=miss
            target = word[i] if i < len(word) else None
            if target and ch == target:
                color = GREEN
            elif target and ch in word:
                color = YELLOW
            else:
                color = RED
            sys.stdout.write(f"{color}{ch}{RESET}")
            sys.stdout.flush()
            motor.observe_token(pop_id)
            current = ch
        else:
            sys.stdout.write(f"{DIM}_{RESET}")
            sys.stdout.flush()

    agent.force_gate_open = False
    motor.babbling_noise = old_noise

    # Score
    n_match = sum(
        1
        for i, ch in enumerate(produced[: len(word)])
        if i < len(word) and ch == word[i]
    )
    match_pct = 100 * n_match / max(len(word), 1)
    print(
        f"\n{DIM}  Target: '{word}' → Got: '{''.join(produced[: len(word)])}' "
        f"({match_pct:.0f}% match){RESET}\n"
    )

    # Reset
    for s in cortex._regions.values():
        s.region.reset_working_memory()


def interactive_loop(
    cortex, encoder, region1, motor, decoder, word_decoder, agent, load_fn
):
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
            elif cmd == "/info":
                print_info(cortex, encoder, region1, motor, decoder)
                continue
            elif cmd == "/reset":
                reset_state(cortex)
                total_bits = 0.0
                n_chars = 0
                recent_bits.clear()
                print(f"{DIM}Working memory cleared.{RESET}")
                continue
            elif cmd == "/stats":
                avg_bpc = total_bits / n_chars if n_chars > 0 else 0
                recent_bpc = sum(recent_bits) / len(recent_bits) if recent_bits else 0
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
                warmup(cortex, encoder, tokens)
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
            elif cmd == "/babble":
                n = int(parts[1]) if len(parts) > 1 else 200
                run_babble(agent, region1, motor, n)
                continue
            elif cmd == "/probe":
                run_probe(cortex, word_decoder)
                continue
            elif cmd == "/echo":
                word = parts[1] if len(parts) > 1 else "the"
                run_echo(agent, motor, word)
                continue
            else:
                print(f"{DIM}Unknown command: {cmd} (try /help){RESET}")
                continue

        # ── INPUT PHASE ──
        # Process each character through full hierarchy.
        # M1 is active but BG should keep gate closed (input phase).
        # Any M1 output = interruption.
        print()
        line_correct = 0
        line_total = 0
        line_bits = []
        line_bursts = []
        last_token = (ord(" "), " ")

        for ch in line:
            token_id = ord(ch)
            last_token = (token_id, ch)

            # S1 prediction BEFORE processing
            preds = decode_prediction(
                region1.l23.active,
                decoder,
                encoder,
            )
            bits = compute_bits(token_id, region1.l23.active, decoder)

            # Train decoder on current state → token
            decoder.observe(token_id, region1.l23.active)

            # Step through full hierarchy
            step_token(agent, token_id, ch)

            # Burst fraction = surprise (per-char, after processing)
            n_active = max(int(region1.active_columns.sum()), 1)
            n_burst = int(region1.bursting_columns.sum())
            burst_frac = n_burst / n_active

            # S2 word decoder: step and check for word boundary
            s2_region = cortex._regions["S2"].region
            completed_word = word_decoder.step(ch, s2_region.l23.firing_rate)

            # Track per-char stats
            line_bursts.append(burst_frac)
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

            # Display: char | surprise | predictions
            color = surprise_color(burst_frac)
            pred_str = format_predictions(preds)
            display_ch = repr(ch) if ch == " " else ch
            surprise_pct = f"{burst_frac:.0%} surprise"

            sys.stdout.write(
                f"  {color}{display_ch:<4s}{RESET}"
                f" {color}{surprise_pct:>12s}{RESET}"
                f"  {DIM}{pred_str}{RESET}\n"
            )

            # At word boundaries, show S2's word-level context
            if completed_word:
                s2_preds = word_decoder.predict(s2_region.l23.firing_rate, k=3)
                if s2_preds:
                    total = max(sum(s for _, s in s2_preds), 1)
                    wp = " ".join(f"{w}:{s / total:.0%}" for w, s in s2_preds)
                    sys.stdout.write(f"  {MAGENTA}     S2 context: {wp}{RESET}\n")

        # ── EOM INJECTION ──
        # Signal turn boundary: M1's turn to speak.
        # Feed EOM token, then neutral input (repeat last char) as speak window.
        step_token(agent, EOM_TOKEN, "")

        # Show transition with summary stats
        if n_chars > 0:
            recent_bpc = sum(recent_bits) / len(recent_bits) if recent_bits else 0
            line_bpc = sum(line_bits) / len(line_bits) if line_bits else 0
            line_acc = line_correct / line_total if line_total > 0 else 0
            avg_burst = sum(line_bursts) / len(line_bursts) if line_bursts else 0
            print(
                f"\n{DIM}  {line_acc:.0%} predicted, "
                f"{avg_burst:.0%} avg surprise"
                f"  [{line_bpc:.1f} bpc]{RESET}"
            )
        print(f"\n{DIM}  [EOM → M1's turn]{RESET}")

        # ── SPEAKING PHASE ──
        # Force BG gate open — we know it's M1's turn.
        # PFC holds context from input phase → goal drive to M1.
        agent.force_gate_open = True
        spoken_chars = []
        silent_steps = 0

        # Snapshot PFC goal and close gate for speaking
        pfc_state = cortex._regions.get("PFC")
        if pfc_state is not None:
            pfc_state.region.snapshot_goal()
            pfc_state.region.gate_open = False

        for _ in range(MAX_SPEAK_STEPS + MAX_SILENT_STEPS):
            # Set PFC goal drive to M1 (if PFC exists)
            if (
                pfc_state is not None
                and hasattr(motor, "_goal_weights")
                and motor._goal_weights is not None
            ):
                motor.set_goal_drive(pfc_state.region.l23.firing_rate)

            # Feed M1's last output as next input (autoregressive)
            step_token(agent, last_token[0], last_token[1])

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
                    sys.stdout.write(f"{DIM}  (M1 silent — gate={gate:.2f}){RESET}")
                    break

        agent.force_gate_open = False

        if spoken_chars:
            sys.stdout.write(f"\n{DIM}  M1 spoke {len(spoken_chars)} chars{RESET}")
        print("\n")

        # Reset working memory for next exchange
        reset_state(cortex)


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
        default="babylm",
        choices=["personachat", "tinydialogues", "babylm"],
        help="Dataset for vocab and warmup (default: babylm)",
    )
    parser.add_argument(
        "--l5-apical",
        action="store_true",
        help="Enable L5 apical segments on sensory regions",
    )
    args = parser.parse_args()

    # Load tokens for vocab (and warmup)
    if args.dataset == "personachat":

        def load_fn(n, **kw):
            return prepare_tokens_personachat(n, **kw)
    elif args.dataset == "tinydialogues":

        def load_fn(n, **kw):
            return prepare_tokens_tinydialogues(n, **kw)
    else:
        from step.data import inject_eom_tokens, prepare_tokens_charlevel

        def load_fn(n, **kw):
            tokens = prepare_tokens_charlevel(n, dataset="babylm")
            return inject_eom_tokens(tokens, segment_length=200)

    # Try to load alphabet from checkpoint first (guarantees dimension match)
    alphabet_str = None
    if args.checkpoint:
        import os
        import pickle

        ckpt_path = args.checkpoint
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.checkpoint}.ckpt")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            if "encoder_alphabet" in ckpt:
                alphabet_str = ckpt["encoder_alphabet"]
                n = len(alphabet_str)
                print(f"{DIM}Loaded alphabet from checkpoint: {n} chars{RESET}")

    if alphabet_str is None:
        # Discover alphabet from corpus
        vocab_size = 1_000_000 if args.checkpoint else 10000
        print(f"{DIM}Loading vocabulary from {args.dataset}...{RESET}")
        sample = load_fn(vocab_size, speak_window=5)
        alphabet = sorted({ch for _, ch in sample if _ >= 0})
        alphabet_str = "".join(alphabet)
        print(f"{DIM}Vocab: {len(alphabet)} chars{RESET}")

    # Build model
    print(f"{DIM}Building model (S1→S2→S3→PFC→M2→M1+BG)...{RESET}")
    cortex, encoder, region1, motor, decoder, word_decoder, agent = build_model(
        alphabet_str, use_l5_apical=args.l5_apical
    )

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
        warmup(cortex, encoder, tokens)

    # Show model info on startup
    print_info(cortex, encoder, region1, motor, decoder)

    # Enter interactive mode
    interactive_loop(
        cortex, encoder, region1, motor, decoder, word_decoder, agent, load_fn
    )
    print(f"{DIM}Goodbye!{RESET}")


if __name__ == "__main__":
    main()
