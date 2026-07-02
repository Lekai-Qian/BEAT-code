"""Unified BEAT vocabulary shared by piano and multi-track modes.

Vocabulary layout (vocab_size = 593):

    [0,   81)    PAT          81 pattern tokens (base-3 codes for τ=4)
    [81,  209)   VEL         128 velocity tokens (0–127);
                              multi-track mode emits constant VEL = DEFAULT_VEL.
    [209, 297)   PIT          88 pitch tokens (descending sort, first absolute
                              then descending intervals; relative encoding).
    [297, 425)   INS         128 instrument tokens (MIDI programs 0–127).
                              Piano: INS+0 = melody, INS+1 = accompaniment.
                              Multi: INS+k = MIDI program k.
    425          BEAT          beat marker
    426          BAR           bar marker
    427          EOS
    428          BOS
    429          PAD
    [430, 438)   TS            8 time-signature slots (see TS_MAP below)
    [438, 453)   TEM          15 tempo bins (40-300 BPM, 20-BPM wide; bin 0 = <40)
    [453, 503)   RESERVED     50 reserved slots for future extension
    503          REST          empty-beat marker (used by both modes)
    504          INS_DRUM      single sentinel for drum program (channel 10);
                              only multi-track uses this.
    [505, 593)   DRUM_PIT     88 absolute pitch tokens for drum tracks;
                              only multi-track uses this.
    VOCAB_SIZE = 593

Design notes:
  - Piano (icml-style) checkpoints trained at the old vocab_size=504 are
    compatible: tokens [0, 504) are unchanged. After loading, expand the
    embed_tokens + lm_head with HF `resize_token_embeddings(593)` — the new
    rows (504..592) are randomly initialized and only used by multi-track.
  - Multi-track tokenizers must be retrained against this layout: the
    previous Ours_multi vocabulary used different offsets entirely.
  - Both modes share PAT / VEL / PIT / INS / BEAT / BAR / EOS / BOS / PAD /
    TS / TEM / REST so the model can in principle handle either mode under
    one set of embeddings.
"""

from dataclasses import dataclass
from typing import Dict


# Time-signature mapping (8 slots, indices map to TS_OFFSET + idx).
TS_MAP: Dict[str, int] = {
    '2/2': 0, '2/4': 1, '3/4': 2, '3/8': 3,
    '4/4': 4, '6/8': 5, '9/8': 6, 'UNK': 7,
}

# Tempo bucketing: 15 bins of width 20 BPM, starting at 40 (bin 0 = <40, bin 14 = >=300).
TEM_BIN_WIDTH = 20
TEM_BIN_START = 40
TEM_NUM_BINS = 15

# Pattern grid:  τ = 4 sixteenth-note steps per beat → 3^τ = 81 PAT codes.
PATTERN_STEPS = 4
PATTERN_VOCAB = 3 ** PATTERN_STEPS  # = 81

NUM_PITCHES = 88                    # piano range MIDI 21..108
NUM_INS = 128                       # GM programs 0..127 (drums handled separately)
NUM_VELOCITIES = 128                # MIDI velocity 0..127

DEFAULT_VEL = 64                    # sentinel velocity for missing-velocity data


@dataclass(frozen=True)
class UnifiedVocab:
    """All token-id offsets and counts.

    Frozen so accidental in-place edits don't drift between modules.
    Instances are interchangeable — all callers should use `VOCAB`.
    """

    pat_offset: int = 0
    pat_vocab_size: int = PATTERN_VOCAB                     # 81

    vel_offset: int = PATTERN_VOCAB                          # 81
    vel_vocab_size: int = NUM_VELOCITIES                     # 128

    pit_offset: int = PATTERN_VOCAB + NUM_VELOCITIES         # 209
    num_pitches: int = NUM_PITCHES                           # 88

    ins_offset: int = PATTERN_VOCAB + NUM_VELOCITIES + NUM_PITCHES  # 297
    num_ins: int = NUM_INS                                   # 128

    beat_token: int = 425
    bar_token: int = 426
    eos_token: int = 427
    bos_token: int = 428
    pad_token: int = 429

    ts_offset: int = 430                                     # 8 slots [430, 438)
    tem_offset: int = 438                                    # 15 slots [438, 453)
    reserved_offset: int = 453                               # 50 slots [453, 503)
    rest_token: int = 503

    ins_drum_token: int = 504                                # multi only
    drum_pit_offset: int = 505                               # multi only [505, 593)

    vocab_size: int = 593

    # State codes for the piano-roll input (matches paper convention).
    state_silence: int = 0
    state_onset: int = 1
    state_sustain: int = 2

    pattern_steps: int = PATTERN_STEPS                       # τ
    default_vel: int = DEFAULT_VEL                           # sentinel

    def ts_to_token(self, ts_str: str) -> int:
        """Time-signature string ('4/4', etc.) → token id."""
        return self.ts_offset + TS_MAP.get(ts_str, TS_MAP['UNK'])

    def tempo_to_token(self, bpm: int) -> int:
        """Raw BPM → TEM token id (15 bins, 20 BPM each, start at 40)."""
        if bpm < TEM_BIN_START:
            bin_idx = 0
        else:
            bin_idx = min((bpm - TEM_BIN_START) // TEM_BIN_WIDTH + 1, TEM_NUM_BINS - 1)
        return self.tem_offset + bin_idx

    def token_to_tempo(self, token: int) -> int:
        """TEM token id → representative BPM (bin midpoint)."""
        bin_idx = token - self.tem_offset
        if bin_idx == 0:
            return TEM_BIN_START - 10
        return TEM_BIN_START + (bin_idx - 1) * TEM_BIN_WIDTH + TEM_BIN_WIDTH // 2

    # -- token-type checks (offset-aware predicates) ---------------------

    def is_pat(self, t: int) -> bool:
        return self.pat_offset <= t < self.pat_offset + self.pat_vocab_size

    def is_vel(self, t: int) -> bool:
        return self.vel_offset <= t < self.vel_offset + self.vel_vocab_size

    def is_pit(self, t: int) -> bool:
        return self.pit_offset <= t < self.pit_offset + self.num_pitches

    def is_ins(self, t: int) -> bool:
        return self.ins_offset <= t < self.ins_offset + self.num_ins

    def is_drum_pit(self, t: int) -> bool:
        return self.drum_pit_offset <= t < self.drum_pit_offset + self.num_pitches

    def is_ts(self, t: int) -> bool:
        return self.ts_offset <= t < self.ts_offset + len(TS_MAP)

    def is_tem(self, t: int) -> bool:
        return self.tem_offset <= t < self.tem_offset + TEM_NUM_BINS


# Module-level singleton — import this everywhere.
VOCAB = UnifiedVocab()


# Convenience re-exports (avoid `VOCAB.x` clutter at call sites).
PAT_OFFSET = VOCAB.pat_offset
PAT_VOCAB_SIZE = VOCAB.pat_vocab_size
VEL_OFFSET = VOCAB.vel_offset
VEL_VOCAB_SIZE = VOCAB.vel_vocab_size
PIT_OFFSET = VOCAB.pit_offset
INS_OFFSET = VOCAB.ins_offset
BEAT_TOKEN = VOCAB.beat_token
BAR_TOKEN = VOCAB.bar_token
EOS_TOKEN = VOCAB.eos_token
BOS_TOKEN = VOCAB.bos_token
PAD_TOKEN = VOCAB.pad_token
TS_OFFSET = VOCAB.ts_offset
TEM_OFFSET = VOCAB.tem_offset
RESERVED_OFFSET = VOCAB.reserved_offset
REST_TOKEN = VOCAB.rest_token
INS_DRUM_TOKEN = VOCAB.ins_drum_token
DRUM_PIT_OFFSET = VOCAB.drum_pit_offset
VOCAB_SIZE = VOCAB.vocab_size

STATE_SILENCE = VOCAB.state_silence
STATE_ONSET = VOCAB.state_onset
STATE_SUSTAIN = VOCAB.state_sustain
