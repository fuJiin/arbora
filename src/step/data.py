from collections.abc import Iterator

from datasets import load_dataset
from transformers import AutoTokenizer

from step.config import EncoderConfig, TrainingConfig
from step.sdr import encode_token


def token_stream(
    training_config: TrainingConfig, encoder_config: EncoderConfig
) -> Iterator[tuple[int, frozenset[int]]]:
    tokenizer = AutoTokenizer.from_pretrained(encoder_config.model_name)
    assert tokenizer is not None
    dataset = load_dataset(
        training_config.dataset_name,
        streaming=True,
        split=training_config.dataset_split,
    )

    t = 0
    for example in dataset:
        token_ids = tokenizer.encode(example["text"])
        for tid in token_ids:
            yield t, encode_token(tid, encoder_config)
            t += 1
            if t >= training_config.max_tokens:
                return
