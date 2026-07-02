"""Model + training configuration for the unified BEAT release.

Both piano and multi-track modes share the same `BackboneModelConfig` — the
backbone is a 16-layer / 768-dim LLaMA over the unified 593-token vocabulary.
Mode-specific dataset paths and hyper-parameters live in `*TrainConfig`.
"""

import os
from dataclasses import dataclass
from typing import Literal

from beat.vocab import (
    VOCAB,
    VOCAB_SIZE,
    PATTERN_STEPS,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)


# ============================================================
# Model architecture (shared across modes)
# ============================================================

@dataclass
class BackboneModelConfig:
    """LLaMA backbone hyper-parameters for the BEAT model."""

    vocab_size: int = VOCAB_SIZE  # 593 (unified)
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 6
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.1

    pad_token_id: int = PAD_TOKEN
    bos_token_id: int = BOS_TOKEN
    eos_token_id: int = EOS_TOKEN

    pattern_steps: int = PATTERN_STEPS

    # max tokens per sample at training time
    train_cutoff_len: int = 2048


# ============================================================
# Training hyper-parameters (per mode)
# ============================================================

@dataclass
class PianoTrainConfig:
    """Piano training config — used by `scripts/train_piano.py`."""
    data_dir: str = os.environ.get("BEAT_PIANO_DATA_DIR", "data/piano_npz")  # 24-tick piano NPZ dir
    output_dir: str = "checkpoints/piano"
    log_dir: str = "logs/piano"

    num_epochs: int = 50
    batch_size: int = 16
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    eval_every_epoch: float = 0.25
    save_every_n_epochs: float = 2.0
    log_every_n_steps: int = 50
    num_workers: int = 16
    eval_split_ratio: float = 0.10
    test_split_ratio: float = 0.10
    random_seed: int = 42


@dataclass
class MultitrackTrainConfig:
    """Multi-track training config — used by `scripts/train_multitrack.py`."""
    data_dir: str = os.environ.get("BEAT_MULTITRACK_DATA_DIR", "data/multitrack_npz")  # multi-track NPZ dir
    output_dir: str = "checkpoints/multitrack"
    log_dir: str = "logs/multitrack"

    num_epochs: int = 30
    batch_size: int = 8
    gradient_accumulation_steps: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    eval_every_epoch: float = 0.25
    save_every_n_epochs: float = 2.0
    log_every_n_steps: int = 50
    num_workers: int = 16
    eval_split_ratio: float = 0.10
    test_split_ratio: float = 0.10
    random_seed: int = 42
