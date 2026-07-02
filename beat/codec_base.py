"""Shared BEAT codec utilities — base class for piano and multi-track tokenizers.

The base class holds:
  - the active `UnifiedVocab` and the τ pattern grid;
  - pure-math helpers shared by both modes (base-3 pattern math, relative pitch
    encoding/decoding with configurable sort direction);
  - token-type predicates that go through the vocab.

What it *doesn't* do: file I/O, NPZ shape handling, drum vs. melodic logic,
velocity sourcing. Subclasses implement those because they differ structurally
between the two modes (piano NPZ is (6, 88, T), multi NPZ is (2N, 88, T); piano
has real velocity, multi emits a constant sentinel; multi has drum tracks with
absolute pitch). See piano/tokenizer.py and multitrack/tokenizer.py for the
concrete pipelines.
"""

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .vocab import VOCAB, UnifiedVocab


class BeatTokenizerBase:
    """Vocab-aware base for BEAT tokenizers.

    Args:
        vocab: a `UnifiedVocab` instance (defaults to the module singleton).
    """

    def __init__(self, vocab: UnifiedVocab = VOCAB):
        self.vocab = vocab
        # cached base-3 weight vector for vectorized pattern (de)coding
        self._base3_weights = (3 ** np.arange(vocab.pattern_steps - 1, -1, -1)).astype(np.int64)

    # =========================================================================
    # base-3 pattern math (shared between both modes)
    # =========================================================================

    def pattern_to_id(self, states: Sequence[int]) -> int:
        """Encode a length-τ tri-state vector → integer in [0, 3^τ)."""
        code = 0
        for s in states:
            code = code * 3 + int(s)
        return code

    def id_to_pattern(self, pat_id: int) -> np.ndarray:
        """Inverse: integer → length-τ tri-state vector."""
        out = np.zeros(self.vocab.pattern_steps, dtype=np.uint8)
        v = int(np.clip(pat_id, 0, self.vocab.pat_vocab_size - 1))
        for i in range(self.vocab.pattern_steps - 1, -1, -1):
            out[i] = v % 3
            v //= 3
        return out

    def encode_pat_matrix(self, tri_state: np.ndarray) -> np.ndarray:
        """Vectorized: (..., τ) tri-state matrix → (...,) PAT id array.

        Faster than a Python loop when encoding many patterns at once.
        """
        return np.asarray(tri_state, dtype=np.int64) @ self._base3_weights

    def decode_pat_matrix(self, pat_ids: np.ndarray) -> np.ndarray:
        """Inverse: PAT id matrix → tri-state (..., τ) digits."""
        return (np.asarray(pat_ids, dtype=np.int64)[..., None] // self._base3_weights) % 3

    # =========================================================================
    # relative pitch encoding (parameterized by sort direction)
    # =========================================================================

    def encode_pit_relative(
        self,
        pitches: Sequence[int],
        *,
        descending: bool = True,
    ) -> List[int]:
        """Encode active pitches as PIT tokens via the paper's relative scheme.

        With `descending=True` (paper / piano convention):
            sort pitches high-to-low → d_1 = p_max (absolute),
            d_j = p_{j-1} − p_j for j ≥ 2 (positive descending intervals).

        With `descending=False`:
            sort low-to-high → d_1 = p_min, d_j = p_j − p_{j-1} (positive
            ascending intervals). Mathematically equivalent but a different
            tokenization; provided so legacy callers can opt in.
        """
        ordered = sorted(pitches, reverse=descending)
        out: List[int] = []
        prev = None
        for p in ordered:
            if prev is None:
                d = p
            elif descending:
                d = prev - p
            else:
                d = p - prev
            if not (0 <= d < self.vocab.num_pitches):
                raise ValueError(f"invalid relative pitch code d={d} (descending={descending})")
            out.append(self.vocab.pit_offset + d)
            prev = p
        return out

    def decode_pit_relative(
        self,
        pit_tokens: Iterable[int],
        *,
        descending: bool = True,
    ) -> List[int]:
        """Inverse of `encode_pit_relative`. Returns absolute pitch indices."""
        out: List[int] = []
        prev = None
        for tok in pit_tokens:
            d = int(tok) - self.vocab.pit_offset
            if prev is None:
                p = d
            elif descending:
                p = prev - d
            else:
                p = prev + d
            out.append(p)
            prev = p
        return out

    # =========================================================================
    # velocity quantization (used by piano; multi may emit a constant sentinel)
    # =========================================================================

    def velocity_to_token(self, velocity: int) -> int:
        """MIDI velocity (0..127) → VEL token id."""
        return self.vocab.vel_offset + int(np.clip(velocity, 0, self.vocab.vel_vocab_size - 1))

    def token_to_velocity(self, token: int) -> int:
        """Inverse of `velocity_to_token`."""
        return int(np.clip(token - self.vocab.vel_offset, 0, 127))

    # =========================================================================
    # abstract — subclasses must implement
    # =========================================================================

    def encode_file(self, npz_path: str, **kwargs) -> List[int]:
        raise NotImplementedError("subclass must implement encode_file(npz_path)")

    def decode_tokens(self, tokens: Sequence[int], **kwargs):
        raise NotImplementedError("subclass must implement decode_tokens(tokens)")
