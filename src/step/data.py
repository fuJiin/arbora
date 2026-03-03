import os
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from step.config import EncoderConfig, TrainingConfig
from step.sdr import encode_token


def _load_hf_token() -> None:
    """Load HF token from .env file if not already in environment."""
    if os.environ.get("HF_TOKEN"):
        return
    # Walk up to find .env
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env_file = parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key in ("HF_TOKEN", "HUGGING_FACE_TOKEN"):
                        os.environ["HF_TOKEN"] = value
                        return
            break


_load_hf_token()


def token_stream(
    training_config: TrainingConfig, encoder_config: EncoderConfig
) -> Iterator[tuple[int, int, frozenset[int]]]:
    """Yield (t, token_id, sdr) tuples from the dataset.

    token_id is needed for SDR definitions and accuracy tracking.
    Uses local cache after first download (no re-streaming).
    """
    tokenizer = AutoTokenizer.from_pretrained(encoder_config.model_name)
    assert tokenizer is not None
    dataset = load_dataset(
        training_config.dataset_name,
        split=training_config.dataset_split,
    )

    t = 0
    for example in dataset:
        token_ids = tokenizer.encode(example["text"])
        for tid in token_ids:
            if tid >= encoder_config.vocab_size:
                tid = 0  # clamp to UNK
            yield t, tid, encode_token(tid, encoder_config)
            t += 1
            if t >= training_config.max_tokens:
                return


def prepare_token_cache(
    training_config: TrainingConfig, encoder_config: EncoderConfig
) -> list[tuple[int, frozenset[int]]]:
    """Download and tokenize once, return cached list of (token_id, sdr).

    Call this once, then use cached_token_stream() to replay for each model.
    """
    cache: list[tuple[int, frozenset[int]]] = []
    for _t, token_id, sdr in token_stream(training_config, encoder_config):
        cache.append((token_id, sdr))
    return cache


def cached_token_stream(
    cache: list[tuple[int, frozenset[int]]],
    max_tokens: int | None = None,
) -> Iterator[tuple[int, int, frozenset[int]]]:
    """Replay a cached token list as (t, token_id, sdr) tuples."""
    limit = max_tokens if max_tokens is not None else len(cache)
    for t, (token_id, sdr) in enumerate(cache[:limit]):
        yield t, token_id, sdr
