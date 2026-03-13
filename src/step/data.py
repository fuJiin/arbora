"""Token loading for cortex experiments.

Loads text datasets, tokenizes with GPT-2, and returns (token_id, token_string)
pairs with STORY_BOUNDARY sentinels between documents.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

STORY_BOUNDARY = -1

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
