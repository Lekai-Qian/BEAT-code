"""Piano BEAT tokenizer — converts (6, 88, T) NPZ piano-roll → token sequence.

Input NPZ layout (per measure, key `measure_k`): shape `(6, 88, T)` with
channels (mel_sus, mel_ons, mel_vel, acc_sus, acc_ons, acc_vel). Velocity is
non-zero MIDI 1..127, sustain/onset are binary.

Encoding (matches paper Algorithm 1, App B.1):
  Each measure → one [BAR] token, then for each beat:
    - active-pitch detection + per-pitch 4-step base-3 PAT,
    - per-pitch mean MIDI VEL,
    - emit [BEAT INS_MEL (PIT, PAT, VEL)+ INS_ACC (PIT, PAT, VEL)+]
      with descending pitch sort,
    - or [REST] if both tracks are empty for that beat.

Uses the shared UnifiedVocab so the resulting token IDs are interchangeable
with multi-track sequences trained against the same vocab.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from beat.codec_base import BeatTokenizerBase
from beat.vocab import VOCAB, UnifiedVocab


# NPZ-side constants (data-side, not vocab-side).
SOURCE_TICKS_PER_BEAT = 24
SUSTAIN_THRESHOLD = 3      # bin counted as sustain only if ≥ 3 ticks active


class PianoTokenizer(BeatTokenizerBase):
    """Piano (2-track: melody + accompaniment) BEAT tokenizer."""

    def __init__(
        self,
        vocab: UnifiedVocab = VOCAB,
        source_ticks_per_beat: int = SOURCE_TICKS_PER_BEAT,
        sustain_threshold: int = SUSTAIN_THRESHOLD,
    ):
        super().__init__(vocab)
        self.source_ticks_per_beat = source_ticks_per_beat
        self.sustain_threshold = sustain_threshold
        self.missing_velocity_count = 0

        # piano track tokens
        self.ins_mel = vocab.ins_offset + 0
        self.ins_acc = vocab.ins_offset + 1

    # ---- per-pitch encoding ------------------------------------------------

    def _bin_bounds(self, length: int, bin_idx: int) -> Tuple[int, int]:
        start = bin_idx * length // self.vocab.pattern_steps
        end = (bin_idx + 1) * length // self.vocab.pattern_steps
        return start, max(end, start + 1)

    def _to_pattern(self, sus: np.ndarray, ons: np.ndarray) -> np.ndarray:
        """One pitch within one beat → length-τ tri-state vector."""
        states = np.full(self.vocab.pattern_steps, self.vocab.state_silence, dtype=np.uint8)
        length = len(sus)
        for j in range(self.vocab.pattern_steps):
            start, end = self._bin_bounds(length, j)
            if int((sus[start:end] > 0).sum()) < self.sustain_threshold:
                continue
            if np.any(ons[start:end] > 0):
                states[j] = self.vocab.state_onset
            else:
                states[j] = self.vocab.state_sustain
        return states

    def _mean_velocity(self, vel: np.ndarray) -> int:
        values = vel[vel > 0]
        if values.size == 0:
            self.missing_velocity_count += 1
            return self.vocab.default_vel
        return int(np.clip(np.rint(values.mean()), 0, 127))

    # ---- track-level encoding ---------------------------------------------

    def _items_to_tokens(self, track_items: List[Tuple[int, int, int]]) -> List[int]:
        """(pitch, pat_id, vel_id) items → flat (PIT, PAT, VEL) triples (descending sort)."""
        ordered = sorted(track_items, key=lambda x: x[0], reverse=True)
        tokens: List[int] = []
        prev_pitch: Optional[int] = None
        for pitch, pat_id, vel_id in ordered:
            d = pitch if prev_pitch is None else prev_pitch - pitch
            if not (0 <= d < self.vocab.num_pitches):
                raise ValueError(f"invalid relative pitch code d={d}")
            tokens.append(self.vocab.pit_offset + d)
            tokens.append(self.vocab.pat_offset + pat_id)
            tokens.append(self.vocab.vel_offset + vel_id)
            prev_pitch = pitch
        return tokens

    # ---- metadata helpers --------------------------------------------------

    def _measure_time_signature(self, data, measure_idx: int, default_ts: str) -> str:
        if 'measure_info' not in data.files:
            return default_ts
        measure_info = data['measure_info']
        if measure_idx >= len(measure_info):
            return default_ts
        try:
            row = measure_info[measure_idx]
            return f"{int(row[0])}/{int(row[1])}"
        except Exception:
            return default_ts

    # ---- main entry --------------------------------------------------------

    def encode_file(self, npz_path: str, pitch_shift: int = 0) -> List[int]:
        """Encode one piano NPZ file into a flat BEAT token sequence."""
        data = np.load(npz_path, allow_pickle=True)
        metadata = data['metadata'].item()
        measure_keys = sorted(
            (k for k in data.files if k.startswith('measure_') and k != 'measure_info'),
            key=lambda x: int(x.split('_')[1]),
        )

        is_first = metadata.get('is_first', True)
        is_last = metadata.get('is_last', True)
        first_ts = self._measure_time_signature(
            data, 0, metadata.get('time_signature', '4/4'),
        )

        tokens: List[int] = []
        if is_first:
            tokens.append(self.vocab.bos_token)
            tokens.append(self.vocab.ts_to_token(first_ts))
            tokens.append(self.vocab.tempo_to_token(int(metadata.get('bpm') or 120)))

        current_ts = first_ts
        for measure_idx, mk in enumerate(measure_keys):
            m = data[mk].copy()  # (6, 88, T) uint8
            measure_ts = self._measure_time_signature(data, measure_idx, current_ts)

            if pitch_shift != 0:
                m = np.roll(m, pitch_shift, axis=1)
                if pitch_shift > 0:
                    m[:, :pitch_shift, :] = 0
                else:
                    m[:, pitch_shift:, :] = 0

            total_ticks = m.shape[2]
            n_beats = (total_ticks + self.source_ticks_per_beat - 1) // self.source_ticks_per_beat
            if n_beats * self.source_ticks_per_beat != total_ticks:
                pad = n_beats * self.source_ticks_per_beat - total_ticks
                m = np.pad(m, ((0, 0), (0, 0), (0, pad)), mode='constant')

            for beat_idx in range(n_beats):
                t0 = beat_idx * self.source_ticks_per_beat
                t1 = t0 + self.source_ticks_per_beat

                if beat_idx == 0:
                    if measure_ts != current_ts:
                        tokens.append(self.vocab.ts_to_token(measure_ts))
                        current_ts = measure_ts
                    tokens.append(self.vocab.bar_token)

                beat_tokens: List[int] = []

                # track 0 = melody (channels 0,1,2); track 1 = accompaniment (3,4,5)
                for track_id, (ch_s, ch_o, ch_v) in (
                    (self.ins_mel, (0, 1, 2)),
                    (self.ins_acc, (3, 4, 5)),
                ):
                    items: List[Tuple[int, int, int]] = []
                    for pitch in range(self.vocab.num_pitches):
                        sus = m[ch_s, pitch, t0:t1]
                        ons = m[ch_o, pitch, t0:t1]
                        if sus.max() == 0 and ons.max() == 0:
                            continue
                        states = self._to_pattern(sus, ons)
                        pat_id = self.pattern_to_id(states)
                        if pat_id == 0:
                            continue
                        vel_id = self._mean_velocity(m[ch_v, pitch, t0:t1])
                        items.append((pitch, pat_id, vel_id))

                    if not items:
                        continue
                    beat_tokens.append(track_id)
                    beat_tokens.extend(self._items_to_tokens(items))

                if beat_tokens:
                    tokens.append(self.vocab.beat_token)
                    tokens.extend(beat_tokens)
                else:
                    tokens.append(self.vocab.rest_token)

        if is_last:
            tokens.append(self.vocab.eos_token)
        return tokens
