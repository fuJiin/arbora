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
from step.config import (
    _default_motor_config,
    _default_pfc_config,
    _default_premotor_config,
    _default_region2_config,
    _default_region3_config,
    _default_s1_config,
    make_motor_region,
    make_pfc_region,
    make_premotor_region,
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
MAX_SPEAK_STEPS = 200  # max chars M1 can generate per turn
MAX_SILENT_STEPS = 10  # give up waiting for M1 after this many silent steps


def surprise_color(bits: float, vocab_size: int = 65) -> str:
    """Color-code by surprise level."""
    random_bpc = math.log2(max(vocab_size, 2))
    if bits < random_bpc * 0.5:
        return GREEN  # Well predicted
    if bits < random_bpc * 0.75:
        return YELLOW  # Somewhat surprising
    return RED  # Very surprising


def build_model(alphabet: str):
    """Build full hierarchy matching staged topology.

    S1→S2→S3 (sensory), PFC (goals), M2 (sequencing), M1 (output).
    Same architecture as cortex_staged.py build_topology().
    """
    encoder = PositionalCharEncoder(alphabet, max_positions=8)

    s1_cfg = _default_s1_config()
    region1 = make_sensory_region(
        s1_cfg,
        encoder.input_dim,
        encoder.encoding_width,
    )

    r2_cfg = _default_region2_config()
    region2 = make_sensory_region(r2_cfg, region1.n_l23_total * 4, seed=123)

    r3_cfg = _default_region3_config()
    region3 = make_sensory_region(r3_cfg, region2.n_l23_total * 8, seed=789)

    # PFC: receives S2 + S3 concatenated
    pfc_cfg = _default_pfc_config()
    pfc = make_pfc_region(
        pfc_cfg, region2.n_l23_total + region3.n_l23_total, seed=999
    )

    # M2: receives S2 + PFC concatenated
    m2_cfg = _default_premotor_config()
    m2 = make_premotor_region(
        m2_cfg, region2.n_l23_total + pfc.n_l23_total, seed=321
    )

    # M1: receives M2 feedforward
    m1_cfg = _default_motor_config()
    m2_n_l23 = m2_cfg.n_columns * m2_cfg.n_l23
    motor = make_motor_region(m1_cfg, m2_n_l23, seed=456)
    # Vocabulary-constrained L5
    output_vocab = [ord(ch) for ch in encoder._char_to_idx]
    motor._output_vocab = np.array(output_vocab, dtype=np.int64)
    motor.n_output_tokens = len(output_vocab)
    n_l23 = motor.n_l23_total
    motor.output_weights = motor._rng.uniform(
        0, 0.01, size=(n_l23, len(output_vocab))
    )
    motor.output_mask = (
        motor._rng.random((n_l23, len(output_vocab))) < 0.5
    ).astype(np.float64)
    motor.output_weights *= motor.output_mask
    motor._output_eligibility = np.zeros((n_l23, len(output_vocab)))

    bg = BasalGanglia(
        context_dim=region1.n_columns + 1,
        learning_rate=0.05,
        seed=789,
    )

    cortex = Topology(encoder, diagnostics_interval=999999)
    cortex.add_region("S1", region1, entry=True)
    cortex.add_region("S2", region2)
    cortex.add_region("S3", region3)
    cortex.add_region("PFC", pfc)
    cortex.add_region("M2", m2)
    cortex.add_region("M1", motor, basal_ganglia=bg)

    # Feedforward
    cortex.connect("S1", "S2", "feedforward", buffer_depth=4, burst_gate=True)
    cortex.connect("S2", "S3", "feedforward", buffer_depth=8, burst_gate=True)
    cortex.connect("S2", "PFC", "feedforward")
    cortex.connect("S3", "PFC", "feedforward")
    cortex.connect("S2", "M2", "feedforward")
    cortex.connect("PFC", "M2", "feedforward")
    cortex.connect("M2", "M1", "feedforward")
    # Surprise
    cortex.connect("S1", "S2", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S2", "S3", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S1", "M1", "surprise", surprise_tracker=SurpriseTracker())
    # Apical — sensory top-down
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S3", "S2", "apical", thalamic_gate=ThalamicGate())
    # Apical — motor monitoring
    cortex.connect("M1", "M2", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("M2", "PFC", "apical", thalamic_gate=ThalamicGate())
    # Apical — cross-hierarchy
    cortex.connect("S1", "M1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())

    decoder = cortex._regions["S1"].dendritic_decoder

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
        f"{c.source}→{c.target}({c.kind})" for c in cortex._connections if c.enabled
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


def run_babble(cortex, encoder, region1, motor, n_chars=200):
    """Watch M1 babble in real time using full topology."""
    print(f"\n{DIM}  M1 babbling {n_chars} chars (PFC→M2→M1)...{RESET}")
    print(f"  {BOLD}", end="", flush=True)

    # Force gate open, set babbling noise for exploration
    old_gate = cortex.force_gate_open
    old_noise = motor.babbling_noise
    cortex.force_gate_open = True
    motor.babbling_noise = 0.3  # Mostly learned policy, some exploration

    current_char = " "
    produced = []
    word_buf = []

    for _i in range(n_chars):
        # Use topology's step() for proper propagation
        token_id = ord(current_char) if current_char else ord(" ")
        cortex.step(token_id, current_char)

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
    cortex.force_gate_open = old_gate
    motor.babbling_noise = old_noise
    print()


def run_probe(cortex):
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

        # Apical gain stats
        if r.has_apical and r._apical_gain_weights is not None:
            gain_mean = float(r._apical_gain_weights.mean())
            gain_max = float(r._apical_gain_weights.max())
            print(f"    Apical gain: mean={gain_mean:.4f} max={gain_max:.3f}")

    print()


def run_echo(cortex, encoder, motor, word: str):
    """Interactive echo: hear a word, then watch M1 try to reproduce it.

    Demonstrates PFC goal maintenance and the full motor pipeline:
    1. Listen: process word through S1→S2→S3→PFC
    2. PFC snapshots goal, gate closes
    3. Speak: M1 produces chars, compare to target
    """
    from step.cortex.pfc import PFCRegion

    # Find PFC
    pfc = None
    for _name, s in cortex._regions.items():
        if isinstance(s.region, PFCRegion):
            pfc = s.region
            break

    print(f"\n{DIM}  Echo mode: listening to '{word}'...{RESET}")

    entry_name = cortex._entry_name
    # Ensure topology is finalized for proper multi-ff propagation
    if not hasattr(cortex, "_ff_conns") or not cortex._ff_conns:
        cortex.finalize()

    # Listen phase — use _propagate_feedforward for multi-ff (PFC)
    if pfc is not None:
        pfc.gate_open = True
    for ch in word:
        encoding = encoder.encode(ch)
        topo_order = cortex._topo_order()
        cortex._propagate_feedforward(topo_order, entry_name, encoding)
        cortex._propagate_signals()

    # PFC snapshot
    if pfc is not None:
        pfc.snapshot_goal()
        pfc.gate_open = False
        print(
            f"{DIM}  PFC goal captured "
            f"(confidence={pfc.confidence:.3f}){RESET}"
        )

    # Speak phase
    print(f"  {DIM}M1 reproducing:{RESET} ", end="", flush=True)
    cortex.force_gate_open = True
    old_noise = motor.babbling_noise
    motor.babbling_noise = 0.2  # Mostly policy, some exploration
    produced = []

    current = " "
    for i in range(len(word) + 2):
        encoding = encoder.encode(current)
        topo_order = cortex._topo_order()
        cortex._propagate_feedforward(topo_order, entry_name, encoding)
        cortex._propagate_signals()

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

    cortex.force_gate_open = False
    motor.babbling_noise = old_noise

    # Score
    n_match = sum(
        1 for i, ch in enumerate(produced[:len(word)])
        if i < len(word) and ch == word[i]
    )
    match_pct = 100 * n_match / max(len(word), 1)
    print(
        f"\n{DIM}  Target: '{word}' → Got: '{''.join(produced[:len(word)])}' "
        f"({match_pct:.0f}% match){RESET}\n"
    )

    # Reset
    for s in cortex._regions.values():
        s.region.reset_working_memory()


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
            elif cmd == "/info":
                print_info(cortex, encoder, region1, motor, decoder)
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
            elif cmd == "/babble":
                n = int(parts[1]) if len(parts) > 1 else 200
                run_babble(cortex, encoder, region1, motor, n)
                continue
            elif cmd == "/probe":
                run_probe(cortex)
                continue
            elif cmd == "/echo":
                word = parts[1] if len(parts) > 1 else "the"
                run_echo(cortex, encoder, motor, word)
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
        last_token = (ord(" "), " ")

        for ch in line:
            token_id = ord(ch)
            last_token = (token_id, ch)

            # S1 prediction BEFORE processing
            preds = decode_prediction(
                region1.active_l23,
                decoder,
                encoder,
            )
            bits = compute_bits(token_id, region1.active_l23, decoder)

            # Train decoder on current state → token
            decoder.observe(token_id, region1.active_l23)

            # Step through full hierarchy
            step_token(cortex, token_id, ch)

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

            # Aligned columns: char | bits | predictions
            sys.stdout.write(
                f"  {color}{display_ch:<4s}{RESET}"
                f" {DIM}{bits:5.1f} bits{RESET}"
                f"  {DIM}{pred_str}{RESET}\n"
            )

        # ── EOM INJECTION ──
        # Signal turn boundary: M1's turn to speak.
        # Feed EOM token, then neutral input (repeat last char) as speak window.
        step_token(cortex, EOM_TOKEN, "")

        # Show transition with summary stats
        if n_chars > 0:
            recent_bpc = sum(recent_bits) / len(recent_bits) if recent_bits else 0
            line_bpc = sum(line_bits) / len(line_bits) if line_bits else 0
            line_acc = line_correct / line_total if line_total > 0 else 0
            burst_pct = float(region1.bursting_columns.sum()) / max(
                region1.n_columns, 1
            )
            print(
                f"\n{DIM}  line: {line_bpc:.2f} bpc, "
                f"{line_acc:.0%} acc, "
                f"{burst_pct:.0%} burst  |  "
                f"overall: {total_bits / n_chars:.2f} bpc "
                f"(recent: {recent_bpc:.2f}){RESET}"
            )
        print(f"\n{DIM}  [EOM → M1's turn]{RESET}")

        # ── SPEAKING PHASE ──
        # Force BG gate open — we know it's M1's turn.
        # PFC holds context from input phase → goal drive to M1.
        cortex.force_gate_open = True
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
                motor.set_goal_drive(pfc_state.region.firing_rate_l23)

            # Feed M1's last output as next input (autoregressive)
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
                    sys.stdout.write(f"{DIM}  (M1 silent — gate={gate:.2f}){RESET}")
                    break

        cortex.force_gate_open = False

        if spoken_chars:
            sys.stdout.write(f"\n{DIM}  M1 spoke {len(spoken_chars)} chars{RESET}")
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
        default="babylm",
        choices=["personachat", "tinydialogues", "babylm"],
        help="Dataset for vocab and warmup (default: babylm)",
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

    # Need enough chars to discover full alphabet (rare chars like digits
    # only appear after many dialogues). 100k when loading checkpoint,
    # smaller sample otherwise since warmup will cover it.
    # Need enough chars to discover full alphabet (rare chars only
    # appear after many documents). Use 1M to match staged training.
    vocab_size = 1_000_000 if args.checkpoint else 10000
    print(f"{DIM}Loading vocabulary from {args.dataset}...{RESET}")
    sample = load_fn(vocab_size, speak_window=5)
    alphabet = sorted({ch for _, ch in sample if _ >= 0})
    alphabet_str = "".join(alphabet)
    print(f"{DIM}Vocab: {len(alphabet)} chars{RESET}")

    # Build model
    print(f"{DIM}Building model (S1→S2→S3→PFC→M2→M1+BG)...{RESET}")
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

    # Show model info on startup
    print_info(cortex, encoder, region1, motor, decoder)

    # Enter interactive mode
    interactive_loop(cortex, encoder, region1, motor, decoder, load_fn)
    print(f"{DIM}Goodbye!{RESET}")


if __name__ == "__main__":
    main()
