"""Dataset for Global Backbone training

Load npz -> BEAT tokenize -> causal-LM training sequences
"""

import os
import warnings
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List

from beat.vocab import BOS_TOKEN, EOS_TOKEN, VOCAB


def _is_prefix_token(t: int) -> bool:
    """BOS / TS / TEM tokens — used as prompt conditioning only, masked from loss.

    Same convention as the multi-track dataset so prompt header tokens
    (BOS + time-signature + tempo) are not counted in the cross-entropy loss.
    """
    return t == BOS_TOKEN or VOCAB.is_ts(t) or VOCAB.is_tem(t)


def split_files(data_dir: str, eval_ratio: float = 0.10,
                test_ratio: float = 0.10, seed: int = 42):
    """Return deterministic train/eval/test file splits."""
    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir) if f.endswith('.npz')
    ])
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_files))
    n_total = len(all_files)

    if n_total == 0:
        return {'train': [], 'eval': [], 'test': []}

    n_eval = max(1, int(n_total * eval_ratio))
    n_test = max(1, int(n_total * test_ratio))
    if n_eval + n_test >= n_total:
        n_eval = min(n_eval, max(0, n_total - 2))
        n_test = min(n_test, max(0, n_total - n_eval - 1))

    eval_idx = indices[:n_eval]
    test_idx = indices[n_eval:n_eval + n_test]
    train_idx = indices[n_eval + n_test:]
    return {
        'train': [all_files[i] for i in train_idx],
        'eval': [all_files[i] for i in eval_idx],
        'test': [all_files[i] for i in test_idx],
    }


class BackboneDataset(Dataset):
    """Load each npz file and tokenize it into a token sequence."""

    def __init__(self, data_dir: str, tokenizer, max_seq_len: int,
                 split: str = 'train', eval_ratio: float = 0.10,
                 test_ratio: float = 0.10,
                 seed: int = 42):
        if split not in ('train', 'eval', 'test'):
            raise ValueError(f"split must be train/eval/test, got {split}")
        splits = split_files(data_dir, eval_ratio=eval_ratio,
                             test_ratio=test_ratio, seed=seed)
        self.files = splits[split]

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self._failed_count = 0
        print(f"BackboneDataset ({split}): {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            # 70% chance to transpose by +/-5 semitones
            pitch_shift = 0
            if self.split == 'train' and np.random.random() < 0.7:
                pitch_shift = np.random.randint(-5, 6)
            tokens = self.tokenizer.encode_file(path, pitch_shift=pitch_shift)
        except Exception as e:
            # on failure return the shortest valid sequence; warn on the first 5, then every 100th
            self._failed_count += 1
            if self._failed_count <= 5 or self._failed_count % 100 == 0:
                warnings.warn(
                    f"[{self.split}] failed to encode #{self._failed_count} "
                    f"({path}): {type(e).__name__}: {e}",
                    stacklevel=2,
                )
            tokens = [BOS_TOKEN, EOS_TOKEN]

        # build labels parallel to tokens; mask BOS/TS/TEM prompt header
        # (same convention as multi-track dataset)
        labels = list(tokens)
        for i, t in enumerate(tokens):
            if _is_prefix_token(t):
                labels[i] = -100
            else:
                break  # prefix region ends at first non-prefix token

        # truncate: keep the head 70% of the time, the tail 30%
        if len(tokens) > self.max_seq_len:
            if np.random.random() < 0.7:
                tokens = tokens[:self.max_seq_len]
                labels = labels[:self.max_seq_len]
            else:
                tokens = tokens[-self.max_seq_len:]
                labels = labels[-self.max_seq_len:]
                # tail-truncated sample lost its prompt header — mask the first
                # 8% so the model isn't asked to predict an out-of-context prefix
                cutoff = int(len(labels) * 0.08)
                labels[:cutoff] = [-100] * cutoff

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


class BackboneCollator:
    """Right-pad variable-length samples to the longest in the batch.

    Expects each sample to be a dict ``{'input_ids', 'labels'}`` (per the
    piano dataset). ``labels`` is preserved verbatim; pad slots are set to
    -100. ``attention_mask`` follows ``input_ids != pad``.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # accept either dict samples or bare tensors (legacy)
        if isinstance(batch[0], torch.Tensor):
            batch = [{'input_ids': t, 'labels': t} for t in batch]

        max_len = max(s['input_ids'].size(0) for s in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, s in enumerate(batch):
            L = s['input_ids'].size(0)
            input_ids[i, :L] = s['input_ids']
            labels[i, :L] = s['labels']
            attention_mask[i, :L] = 1
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
