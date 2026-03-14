"""Token loading for cortex experiments.

Loads text datasets, tokenizes with GPT-2, and returns (token_id, token_string)
pairs with STORY_BOUNDARY sentinels between documents.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

STORY_BOUNDARY = -1
EOM_TOKEN = -2  # End-of-message: signals turn boundary for motor RL

DATASETS = {
    "tinystories": "roneneldan/TinyStories",
    "babylm": "nilq/babylm-10M",
    "tinydialogues": "styfeng/TinyDialogues",
    "personachat": "AlekseyKorshuk/persona-chat",
}


def prepare_tokens(
    max_tokens: int,
    dataset: str = "babylm",
) -> list[tuple[int, str]]:
    """Load and tokenize a text dataset for cortex experiments.

    Args:
        max_tokens: Maximum number of tokens to load.
        dataset: Dataset name — "babylm" or "tinystories".

    Returns:
        List of (token_id, token_string) pairs with STORY_BOUNDARY
        sentinels between documents.
    """
    dataset_path = DATASETS.get(dataset, dataset)
    print(f"Loading {dataset_path}...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset(dataset_path, split="train")

    tokens: list[tuple[int, str]] = []
    t = 0

    if dataset == "babylm":
        # BabyLM uses empty lines as document boundaries
        in_doc = False
        for ex in ds:
            text = ex.get("text", "").strip()
            if not text:
                if in_doc:
                    tokens.append((STORY_BOUNDARY, ""))
                    t += 1
                    in_doc = False
                if t >= max_tokens:
                    break
                continue
            in_doc = True
            for tid in tokenizer.encode(text):
                tokens.append((tid, tokenizer.decode([tid])))
                t += 1
                if t >= max_tokens:
                    break
            if t >= max_tokens:
                break
    else:
        # TinyStories / other: each example is a document
        first = True
        for ex in ds:
            if not first:
                tokens.append((STORY_BOUNDARY, ""))
                t += 1
                if t >= max_tokens:
                    break
            first = False
            for tid in tokenizer.encode(ex["text"]):
                tokens.append((tid, tokenizer.decode([tid])))
                t += 1
                if t >= max_tokens:
                    break
            if t >= max_tokens:
                break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(f"  {len(tokens):,} tokens, {unique} unique, {boundaries + 1} documents")
    return tokens


def prepare_tokens_charlevel(
    max_tokens: int,
    dataset: str = "babylm",
) -> list[tuple[int, str]]:
    """Load text and split into individual characters.

    Each character becomes a token with token_id = ord(char).
    Vocabulary is ~80 printable ASCII characters.

    Returns same format as prepare_tokens(): list of (token_id, token_string)
    with STORY_BOUNDARY sentinels between documents.
    """
    dataset_path = DATASETS.get(dataset, dataset)
    print(f"Loading {dataset_path} (char-level)...")
    ds = load_dataset(dataset_path, split="train")

    tokens: list[tuple[int, str]] = []
    t = 0

    if dataset == "babylm":
        in_doc = False
        for ex in ds:
            text = ex.get("text", "").strip()
            if not text:
                if in_doc:
                    tokens.append((STORY_BOUNDARY, ""))
                    t += 1
                    in_doc = False
                if t >= max_tokens:
                    break
                continue
            in_doc = True
            for ch in text:
                tokens.append((ord(ch), ch))
                t += 1
                if t >= max_tokens:
                    break
            if t >= max_tokens:
                break
    else:
        first = True
        for ex in ds:
            if not first:
                tokens.append((STORY_BOUNDARY, ""))
                t += 1
                if t >= max_tokens:
                    break
            first = False
            for ch in ex["text"]:
                tokens.append((ord(ch), ch))
                t += 1
                if t >= max_tokens:
                    break
            if t >= max_tokens:
                break

    unique = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(f"  {len(tokens):,} chars, {unique} unique, {boundaries + 1} documents")
    return tokens


def inject_eom_tokens(
    tokens: list[tuple[int, str]],
    *,
    segment_length: int = 0,
    speak_window: int = 10,
) -> list[tuple[int, str]]:
    """Insert EOM_TOKEN to create turn boundaries with speaking windows.

    After each EOM, M1 gets `speak_window` steps (repeating the last token
    as neutral input) before STORY_BOUNDARY resets everything. This gives
    the motor cortex time to practice speaking/staying silent.

    Two modes:
    1. Before each STORY_BOUNDARY (always): natural document endings.
    2. Every `segment_length` tokens (if > 0): synthetic turn boundaries
       within documents, so M1 gets frequent practice even in short runs.

    Pattern: ...tokens... <EOM> [speak_window x last_token] <BOUNDARY>
    """
    result: list[tuple[int, str]] = []
    since_last = 0
    last_token = (0, " ")  # fallback

    for tid, tstr in tokens:
        if tid == STORY_BOUNDARY:
            result.append((EOM_TOKEN, ""))
            # Speak window: repeat last token as neutral input
            for _ in range(speak_window):
                result.append(last_token)
            result.append((STORY_BOUNDARY, ""))
            since_last = 0
        else:
            result.append((tid, tstr))
            last_token = (tid, tstr)
            since_last += 1
            if segment_length > 0 and since_last >= segment_length:
                result.append((EOM_TOKEN, ""))
                for _ in range(speak_window):
                    result.append(last_token)
                result.append((STORY_BOUNDARY, ""))
                since_last = 0

    n_eom = sum(1 for tid, _ in result if tid == EOM_TOKEN)
    print(
        f"  Injected {n_eom} EOM tokens "
        f"(segment={segment_length}, window={speak_window})"
    )
    return result


def prepare_tokens_tinydialogues(
    max_tokens: int,
    *,
    speak_window: int = 10,
    split: str = "train",
) -> list[tuple[int, str]]:
    """Load TinyDialogues with real speaker-alternation turn boundaries.

    Each dialogue has turns marked by **SpeakerName**: prefix.
    Child turns become EOM phases (M1 should speak), adult turns are input.

    Turn structure:
      Adult utterance chars → EOM_TOKEN → [speak_window x last_char] →
      Child utterance chars → STORY_BOUNDARY (between dialogues)

    Args:
        max_tokens: Maximum characters to load.
        speak_window: Steps of neutral input after each EOM for M1 practice.
        split: Dataset split ("train" or "validation").

    Returns:
        List of (ord(char), char) pairs with EOM_TOKEN and STORY_BOUNDARY.
    """
    import re

    dataset_path = DATASETS["tinydialogues"]
    print(f"Loading {dataset_path} (char-level, speaker turns)...")
    ds = load_dataset(dataset_path, split=split)

    # Pattern: **SpeakerName**: utterance
    turn_re = re.compile(r"\*\*([^*]+)\*\*:\s*")

    tokens: list[tuple[int, str]] = []
    t = 0
    n_dialogues = 0
    n_eom = 0

    for ex in ds:
        text = ex.get("text", "")
        # Remove end-of-text marker
        text = text.replace("<|endoftext|>", "").strip()
        if not text:
            continue

        # Split into turns
        parts = turn_re.split(text)
        # parts = [pre, speaker1, utterance1, speaker2, utterance2, ...]
        # Skip any pre-speaker text (usually empty)
        turns: list[tuple[str, str]] = []
        i = 1  # skip parts[0] (text before first speaker)
        while i + 1 < len(parts):
            speaker = parts[i].strip()
            utterance = parts[i + 1].strip()
            # Unescape literal \n\n used as line breaks in the dataset
            utterance = utterance.replace("\\n\\n", "\n\n")
            if utterance:
                turns.append((speaker, utterance))
            i += 2

        if not turns:
            continue

        # Add story boundary between dialogues
        if n_dialogues > 0:
            tokens.append((STORY_BOUNDARY, ""))
            t += 1
            if t >= max_tokens:
                break

        # Detect child speaker (heuristic: "Child" in name)
        child_names = {
            s for s, _ in turns if "child" in s.lower() or "kid" in s.lower()
        }

        for speaker, utterance in turns:
            is_child = speaker in child_names

            if is_child:
                # Before child turn: insert EOM (M1's turn to speak)
                tokens.append((EOM_TOKEN, ""))
                n_eom += 1
                # Speak window: repeat last real char as neutral input
                if tokens:
                    last_real = (ord(" "), " ")
                    for tid_prev, ts_prev in reversed(tokens):
                        if tid_prev >= 0:
                            last_real = (tid_prev, ts_prev)
                            break
                    for _ in range(speak_window):
                        tokens.append(last_real)
                        t += 1
                        if t >= max_tokens:
                            break
                if t >= max_tokens:
                    break

            # Emit utterance characters
            for ch in utterance:
                tokens.append((ord(ch), ch))
                t += 1
                if t >= max_tokens:
                    break
            if t >= max_tokens:
                break

        n_dialogues += 1
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid >= 0})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(
        f"  {len(tokens):,} chars, {unique} unique, "
        f"{n_dialogues} dialogues, {boundaries} boundaries, {n_eom} EOM tokens"
    )
    return tokens


def prepare_tokens_personachat(
    max_tokens: int,
    *,
    speak_window: int = 10,
    split: str = "train",
) -> list[tuple[int, str]]:
    """Load PersonaChat with alternating speaker turns.

    Dialogues have two speakers alternating. Odd turns (index 1, 3, ...)
    are treated as "our" responses (EOM phases for M1 to practice).

    Turn structure:
      Speaker A chars → EOM_TOKEN → [speak_window x last_char] →
      Speaker B chars → ... → STORY_BOUNDARY (between dialogues)

    Args:
        max_tokens: Maximum characters to load.
        speak_window: Steps of neutral input after each EOM.
        split: Dataset split.

    Returns:
        List of (ord(char), char) pairs with EOM_TOKEN and STORY_BOUNDARY.
    """
    dataset_path = DATASETS["personachat"]
    print(f"Loading {dataset_path} (char-level, persona chat)...")
    ds = load_dataset(dataset_path, split=split)

    tokens: list[tuple[int, str]] = []
    t = 0
    n_dialogues = 0
    n_eom = 0

    for ex in ds:
        utterances = ex.get("utterances", [])
        if not utterances:
            continue

        # Reconstruct full dialogue from the last utterance's history
        # (it contains all previous turns) + the chosen response
        last_turn = utterances[-1]
        history = last_turn.get("history", [])
        candidates = last_turn.get("candidates", [])
        chosen = candidates[-1] if candidates else ""

        # Full dialogue: history + chosen response
        turns = list(history)
        if chosen:
            turns.append(chosen)

        if len(turns) < 2:
            continue

        # Story boundary between dialogues
        if n_dialogues > 0:
            tokens.append((STORY_BOUNDARY, ""))
            t += 1
            if t >= max_tokens:
                break

        for turn_idx, utterance in enumerate(turns):
            # Odd turns = "our" response → inject EOM before
            if turn_idx > 0 and turn_idx % 2 == 1:
                tokens.append((EOM_TOKEN, ""))
                n_eom += 1
                # Speak window: repeat last real char
                last_real = (ord(" "), " ")
                for tid_prev, ts_prev in reversed(tokens):
                    if tid_prev >= 0:
                        last_real = (tid_prev, ts_prev)
                        break
                for _ in range(speak_window):
                    tokens.append(last_real)
                    t += 1
                    if t >= max_tokens:
                        break
                if t >= max_tokens:
                    break

            # Emit utterance characters
            for ch in utterance:
                tokens.append((ord(ch), ch))
                t += 1
                if t >= max_tokens:
                    break
            if t >= max_tokens:
                break

        n_dialogues += 1
        if t >= max_tokens:
            break

    unique = len({tid for tid, _ in tokens if tid >= 0})
    boundaries = sum(1 for tid, _ in tokens if tid == STORY_BOUNDARY)
    print(
        f"  {len(tokens):,} chars, {unique} unique, "
        f"{n_dialogues} dialogues, {boundaries} boundaries, "
        f"{n_eom} EOM tokens"
    )
    return tokens
