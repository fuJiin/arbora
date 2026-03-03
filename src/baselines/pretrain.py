"""Pre-train MiniGPT on TinyStories."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from baselines.mini_gpt import MiniGPT, MiniGPTConfig


@dataclass
class PretrainConfig:
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    max_tokens: int = 100_000
    batch_size: int = 32
    epochs: int = 3
    lr: float = 1e-3
    warmup_steps: int = 0
    min_lr: float = 0.0


def pretrain_mini_gpt(
    gpt_config: MiniGPTConfig,
    pretrain_config: PretrainConfig,
    checkpoint_path: Path,
) -> Path:
    """Pre-train MiniGPT and save checkpoint. Returns checkpoint path."""
    if checkpoint_path.exists():
        print(f"Checkpoint already exists: {checkpoint_path}")
        return checkpoint_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(gpt_config).to(device)
    print(f"MiniGPT params: {model.param_count():,} | device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_config.lr)

    # Collect tokens from dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset(
        pretrain_config.dataset_name,
        streaming=True,
        split=pretrain_config.dataset_split,
    )

    print(f"Collecting {pretrain_config.max_tokens:,} tokens...")
    all_tokens: list[int] = []
    for example in dataset:
        tokens = tokenizer.encode(example["text"])
        # Clamp tokens >= vocab_size to UNK (token 0)
        tokens = [t if t < gpt_config.vocab_size else 0 for t in tokens]
        all_tokens.extend(tokens)
        if len(all_tokens) >= pretrain_config.max_tokens:
            all_tokens = all_tokens[: pretrain_config.max_tokens]
            break

    print(f"Collected {len(all_tokens):,} tokens")

    # Build sequences of length block_size
    seq_len = gpt_config.block_size
    sequences = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        sequences.append(all_tokens[i : i + seq_len])

    data = torch.tensor(sequences, dtype=torch.long)
    n_seqs = data.size(0)
    print(f"Training sequences: {n_seqs}")

    # Compute total training steps for LR schedule
    bs = pretrain_config.batch_size
    steps_per_epoch = (n_seqs + bs - 1) // bs
    total_steps = steps_per_epoch * pretrain_config.epochs

    # Training loop
    model.train()
    global_step = 0
    for epoch in range(pretrain_config.epochs):
        perm = torch.randperm(n_seqs)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_seqs, pretrain_config.batch_size):
            # Cosine LR schedule with linear warmup
            cfg = pretrain_config
            warmup = cfg.warmup_steps
            if warmup > 0 and global_step < warmup:
                lr = cfg.lr * global_step / warmup
            elif total_steps > warmup:
                progress = (global_step - warmup) / (
                    total_steps - warmup
                )
                lr = cfg.min_lr + 0.5 * (
                    cfg.lr - cfg.min_lr
                ) * (1 + math.cos(math.pi * progress))
            else:
                lr = cfg.lr
            optimizer.param_groups[0]["lr"] = lr

            batch_idx = perm[i : i + pretrain_config.batch_size]
            batch = data[batch_idx].to(device)

            logits = model(batch)
            # Predict next token from each position
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, gpt_config.vocab_size),
                batch[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{pretrain_config.epochs}: loss = {avg_loss:.4f}")

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path
