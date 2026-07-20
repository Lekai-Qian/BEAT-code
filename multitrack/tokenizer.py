"""Multi-track BEAT tokenizer — converts (2*N, 88, T) NPZ piano-roll → token sequence.

Input NPZ layout (per measure, key `measure_k`): shape `(2*num_tracks, 88, T)`.
Channels per track are (sustain, onset), binary 0/1 — the original Ours_multi
preprocessing did not retain MIDI velocity. T is already aligned to the BEAT
grid (T is a multiple of τ = 4).

Encoding (unified-vocab convention, paper-aligned):
  For each measure → [BAR] then for each beat:
    - if all tracks empty for this beat → [REST]
    - else → [BEAT] then for each non-empty track:
        - melodic tracks: [INS_program (PIT, PAT, VEL)+] with descending pitch
          sort (paper App A.2 ablation).
        - drum tracks (track program == 128): [INS_DRUM (DRUM_PIT, PAT, VEL)+]
          using ABSOLUTE pitch in DRUM_PIT space (intervals are meaningless
          for drum pitches).
    - VEL is always `vocab.default_vel` (= 64) for now — multi data has no
      real velocity. The token slot is reserved so multi sequences share
      structure with piano sequences and can use the same backbone vocab.

Notable changes from the original Ours_multi implementation:
  - DESCENDING pitch sort instead of ascending (paper convention).
  - Empty beats emit `REST` instead of bare `INS` markers.
  - Velocity is always emitted as a constant token (formerly omitted).
  - `INS_DRUM` is a single dedicated sentinel at id 504 instead of program 128
    encoded inside the INS range.
  - Tempo encoded with 15 fine-grained bins instead of 4 coarse buckets.
  - Token IDs all shifted to the UnifiedVocab layout.
"""

from typing import List, Sequence, Tuple

import numpy as np

from beat.codec_base import BeatTokenizerBase
from beat.vocab import VOCAB, UnifiedVocab


DRUM_PROGRAM = 128  # GM channel-10 sentinel


class MultitrackTokenizer(BeatTokenizerBase):
    """Multi-track BEAT tokenizer (arbitrary MIDI programs + drums, no velocity)."""

    def __init__(self, vocab: UnifiedVocab = VOCAB):
        super().__init__(vocab)
        self.vel_token = vocab.vel_offset + vocab.default_vel  # constant VEL sentinel

    # ---- per-beat per-track encoding --------------------------------------

    def _encode_beat_track(
        self,
        beat: np.ndarray,
        program: int,
    ) -> List[int]:
        """Encode one (2, 88, τ) track segment for one beat into a list of tokens.

        Returns [] if no active pitches.
        """
        sustain = beat[0]   # (88, τ)
        onset = beat[1] * (sustain > 0)
        # tri-state matrix (88, τ) using the paper / piano convention:
        #   0 = silence, 1 = STATE_ONSET (attack), 2 = STATE_SUSTAIN (continuation).
        # NOTE: this differs from the legacy Ours_multi which used sustain+onset
        # (where 1=sustain, 2=attack). PAT ids produced here therefore index into
        # the SAME musical patterns as the piano tokenizer — required for
        # piano/multi to share a single backbone vocabulary.
        sustain_i = sustain.astype(np.int64)
        onset_i = onset.astype(np.int64)
        combined = np.where(
            onset_i > 0, self.vocab.state_onset,
            np.where(sustain_i > 0, self.vocab.state_sustain, self.vocab.state_silence),
        ).astype(np.int64)
        pat_ids = self.encode_pat_matrix(combined)  # (88,)
        active = np.flatnonzero(pat_ids)
        if active.size == 0:
            return []

        is_drum = (program == DRUM_PROGRAM)
        ins_token = self.vocab.ins_drum_token if is_drum else (self.vocab.ins_offset + program)
        out: List[int] = [ins_token]

        if is_drum:
            # absolute pitch — drums have no meaningful intervals
            for pitch_idx in active:
                out.append(self.vocab.drum_pit_offset + int(pitch_idx))
                out.append(self.vocab.pat_offset + int(pat_ids[pitch_idx]))
                out.append(self.vel_token)
        else:
            # descending sort + relative intervals (paper convention)
            sorted_pitches = sorted(active.tolist(), reverse=True)
            prev = None
            for p in sorted_pitches:
                d = p if prev is None else (prev - p)
                if not (0 <= d < self.vocab.num_pitches):
                    raise ValueError(f"invalid relative pitch d={d} for pitches {sorted_pitches}")
                out.append(self.vocab.pit_offset + d)
                out.append(self.vocab.pat_offset + int(pat_ids[p]))
                out.append(self.vel_token)
                prev = p
        return out

    # ---- main entry --------------------------------------------------------

    def encode_file(self, npz_path: str) -> List[int]:
        """Encode one multi-track NPZ file into a flat BEAT token sequence."""
        data = np.load(npz_path, allow_pickle=True)
        metadata = data['metadata'].item()
        num_tracks = int(metadata.get('num_tracks', 0))
        instruments: Sequence[int] = list(metadata.get('instruments', []))
        if len(instruments) != num_tracks:
            raise ValueError(
                f"{npz_path}: instruments ({len(instruments)}) != num_tracks ({num_tracks})"
            )

        measure_keys = sorted(
            (k for k in data.files if k.startswith('measure_')),
            key=lambda x: int(x.split('_')[1]),
        )
        is_continuation = bool(metadata.get('is_continuation', False))
        bpm = int(metadata.get('bpm', 120) or 120)
        # Index 0 is the valid 4/4 slot, so do not use ``or`` for fallback.
        raw_ts_idx = metadata.get('time_signature_idx')
        ts_idx = 0 if raw_ts_idx is None else int(raw_ts_idx)
        ts_str = _ts_from_idx(ts_idx)

        tokens: List[int] = []
        if not is_continuation:
            tokens.append(self.vocab.bos_token)
            tokens.append(self.vocab.ts_to_token(ts_str))
            tokens.append(self.vocab.tempo_to_token(bpm))

        beat_length = self.vocab.pattern_steps  # τ = 4 time steps per beat

        for measure_idx, mk in enumerate(measure_keys):
            m = data[mk]  # (2*num_tracks, 88, T)
            T = m.shape[2]
            # pad to multiple of τ
            pad = (-T) % beat_length
            if pad:
                m = np.pad(m, ((0, 0), (0, 0), (0, pad)), mode='constant')
            num_beats = m.shape[2] // beat_length

            tokens.append(self.vocab.bar_token)
            for b in range(num_beats):
                t0, t1 = b * beat_length, (b + 1) * beat_length
                beat_view = m[:, :, t0:t1]  # (2*N, 88, τ)

                beat_token_block: List[int] = []
                for track_idx, program in enumerate(instruments):
                    track_seg = beat_view[track_idx * 2: track_idx * 2 + 2]  # (2, 88, τ)
                    beat_token_block.extend(self._encode_beat_track(track_seg, int(program)))

                if beat_token_block:
                    tokens.append(self.vocab.beat_token)
                    tokens.extend(beat_token_block)
                else:
                    tokens.append(self.vocab.rest_token)

        if not is_continuation:
            tokens.append(self.vocab.eos_token)
        return tokens


# ---- helpers ---------------------------------------------------------------

# Ours_multi's metadata uses an index into a fixed TS list. Mapping below
# follows the preprocessing convention in `Ours_multi/data_prep/preprocess_xml.py`.
_TS_LIST = ['4/4', '3/4', '2/4', '6/8', '2/2']


def _ts_from_idx(idx: int) -> str:
    if 0 <= idx < len(_TS_LIST):
        return _TS_LIST[idx]
    return 'UNK'
