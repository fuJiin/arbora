from datasets import load_dataset


def get_tiny_stories_stream():
    dataset = load_dataset("roneneldan/TinyStories", streaming=True, split="train")
    for story in dataset:
        yield story["text"]

