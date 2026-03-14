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
