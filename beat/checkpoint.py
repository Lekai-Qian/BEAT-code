"""Shared checkpoint path validation helpers."""

import os


def require_checkpoint_file(checkpoint_path: str) -> str:
    """Return a valid checkpoint path or fail before building a random model."""
    if not checkpoint_path:
        raise ValueError("checkpoint path must not be empty")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return checkpoint_path
