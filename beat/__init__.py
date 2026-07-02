"""Unified BEAT library — shared vocab, base codec, model wrapper."""

from .vocab import (
    VOCAB,
    UnifiedVocab,
    TS_MAP,
    TEM_BIN_WIDTH,
    TEM_BIN_START,
    TEM_NUM_BINS,
    PATTERN_STEPS,
    PATTERN_VOCAB,
    NUM_PITCHES,
    NUM_INS,
    NUM_VELOCITIES,
    DEFAULT_VEL,
    PAT_OFFSET, PAT_VOCAB_SIZE,
    VEL_OFFSET, VEL_VOCAB_SIZE,
    PIT_OFFSET,
    INS_OFFSET,
    BEAT_TOKEN, BAR_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN,
    TS_OFFSET, TEM_OFFSET, RESERVED_OFFSET, REST_TOKEN,
    INS_DRUM_TOKEN, DRUM_PIT_OFFSET, VOCAB_SIZE,
    STATE_SILENCE, STATE_ONSET, STATE_SUSTAIN,
)

from .codec_base import BeatTokenizerBase

__all__ = [
    "VOCAB",
    "UnifiedVocab",
    "BeatTokenizerBase",
    "TS_MAP",
    "VOCAB_SIZE",
    "PAT_OFFSET", "PAT_VOCAB_SIZE",
    "VEL_OFFSET", "VEL_VOCAB_SIZE",
    "PIT_OFFSET",
    "INS_OFFSET",
    "BEAT_TOKEN", "BAR_TOKEN", "EOS_TOKEN", "BOS_TOKEN", "PAD_TOKEN",
    "TS_OFFSET", "TEM_OFFSET", "REST_TOKEN",
    "INS_DRUM_TOKEN", "DRUM_PIT_OFFSET",
    "STATE_SILENCE", "STATE_ONSET", "STATE_SUSTAIN",
    "DEFAULT_VEL",
]
