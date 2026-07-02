"""Token sequence decoder: tokens -> pianoroll -> npz -> MIDI."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from beat.vocab import (
    BAR_TOKEN,
    BEAT_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    PAT_OFFSET,
    PAT_VOCAB_SIZE,
    PATTERN_STEPS,
    PIT_OFFSET,
    NUM_PITCHES,
    REST_TOKEN,
    STATE_ONSET,
    STATE_SILENCE,
    STATE_SUSTAIN,
    INS_OFFSET,
    TS_MAP,
    TS_OFFSET,
    TEM_OFFSET,
    TEM_BIN_START,
    TEM_BIN_WIDTH,
    TEM_NUM_BINS,
    VEL_OFFSET,
    VEL_VOCAB_SIZE,
)

# piano-only constants (not part of the shared vocab)
SOURCE_TICKS_PER_BEAT = 24
PITCH_ENCODING_RELATIVE = "relative"
PITCH_ENCODING_ABSOLUTE = "absolute"
PITCH_ENCODING = PITCH_ENCODING_RELATIVE


@dataclass
class BeatData:
    """Parsed contents for one beat."""

    is_bar_start: bool = False
    time_signature: Optional[str] = None
    tracks: Dict[int, List[Tuple[int, int, int]]] = field(default_factory=dict)
    # tracks[track_id] = [(pitch_idx, pattern_id, velocity_id), ...]


@dataclass
class ParsedSequence:
    """Structured token sequence."""

    time_signature: str = '4/4'
    bpm: int = 120
    beats: List[BeatData] = field(default_factory=list)


def _finish_beat(result: ParsedSequence, beat: Optional[BeatData]):
    if beat is not None:
        result.beats.append(beat)


def parse_tokens(tokens: List[int], pitch_encoding: str = PITCH_ENCODING) -> ParsedSequence:
    """Parse flat tokens into beat/track/pitch-pattern-velocity data."""
    if pitch_encoding not in (PITCH_ENCODING_ABSOLUTE, PITCH_ENCODING_RELATIVE):
        raise ValueError(f"unsupported pitch_encoding: {pitch_encoding}")
    result = ParsedSequence()
    inv_ts_map = {v: k for k, v in TS_MAP.items()}
    i = 0
    current_beat = None
    current_track = None
    current_track_prev_pitch = None
    current_beat_is_rest = False
    current_time_signature = result.time_signature

    while i < len(tokens):
        token = tokens[i]

        if token in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
            i += 1
            continue

        if TS_OFFSET <= token < TS_OFFSET + len(TS_MAP):
            current_time_signature = inv_ts_map.get(token - TS_OFFSET, '4/4')
            if current_beat is None and not result.beats:
                result.time_signature = current_time_signature
            i += 1
            continue

        if TEM_OFFSET <= token < TEM_OFFSET + TEM_NUM_BINS:
            bin_idx = token - TEM_OFFSET
            if bin_idx == 0:
                result.bpm = TEM_BIN_START - 10
            else:
                result.bpm = TEM_BIN_START + (bin_idx - 1) * TEM_BIN_WIDTH + TEM_BIN_WIDTH // 2
            i += 1
            continue

        if token == BAR_TOKEN:
            _finish_beat(result, current_beat)
            current_beat = BeatData(
                is_bar_start=True,
                time_signature=current_time_signature,
            )
            current_track = None
            current_track_prev_pitch = None
            current_beat_is_rest = False
            i += 1
            continue

        if token == REST_TOKEN:
            if current_beat is None:
                current_beat = BeatData(is_bar_start=False)
            elif current_beat.tracks or current_beat_is_rest:
                _finish_beat(result, current_beat)
                current_beat = BeatData(is_bar_start=False)
            current_beat_is_rest = True
            current_track = None
            current_track_prev_pitch = None
            i += 1
            continue

        if token == BEAT_TOKEN:
            if (
                current_beat is not None
                and not (current_beat.is_bar_start and not current_beat.tracks and not current_beat_is_rest)
            ):
                _finish_beat(result, current_beat)
                current_beat = BeatData(is_bar_start=False)
            elif current_beat is None:
                current_beat = BeatData(is_bar_start=False)
            current_beat_is_rest = False
            current_track = None
            current_track_prev_pitch = None
            i += 1
            continue

        if INS_OFFSET <= token < INS_OFFSET + 128:
            current_track = token - INS_OFFSET
            current_track_prev_pitch = None
            current_beat_is_rest = False
            if current_beat is not None:
                current_beat.tracks.setdefault(current_track, [])
            i += 1
            continue

        if (
            PIT_OFFSET <= token < PIT_OFFSET + NUM_PITCHES
            and i + 2 < len(tokens)
            and PAT_OFFSET <= tokens[i + 1] < PAT_OFFSET + PAT_VOCAB_SIZE
            and VEL_OFFSET <= tokens[i + 2] < VEL_OFFSET + VEL_VOCAB_SIZE
        ):
            pitch_code = token - PIT_OFFSET
            if pitch_encoding == PITCH_ENCODING_RELATIVE:
                if current_track_prev_pitch is None:
                    pitch = pitch_code
                else:
                    pitch = current_track_prev_pitch - pitch_code
            else:
                pitch = pitch_code
            pattern_id = tokens[i + 1] - PAT_OFFSET
            velocity_id = tokens[i + 2] - VEL_OFFSET
            if current_beat is not None and current_track is not None and 0 <= pitch < NUM_PITCHES:
                current_beat.tracks.setdefault(current_track, [])
                current_beat.tracks[current_track].append((pitch, pattern_id, velocity_id))
                current_track_prev_pitch = pitch
            i += 3
            continue

        i += 1

    _finish_beat(result, current_beat)

    return result


class BeatPatternDecoder:
    """Decode 4-step base-3 BEAT patterns to 24-tick pianoroll rows."""

    def __init__(
        self,
        pattern_steps: int = PATTERN_STEPS,
        source_ticks_per_beat: int = SOURCE_TICKS_PER_BEAT,
    ):
        self.pattern_steps = pattern_steps
        self.source_ticks_per_beat = source_ticks_per_beat

    def _bin_bounds(self, bin_idx: int) -> tuple[int, int]:
        start = bin_idx * self.source_ticks_per_beat // self.pattern_steps
        end = (bin_idx + 1) * self.source_ticks_per_beat // self.pattern_steps
        return start, end

    def pattern_to_states(self, pattern_id: int) -> np.ndarray:
        states = np.zeros(self.pattern_steps, dtype=np.uint8)
        value = int(np.clip(pattern_id, 0, PAT_VOCAB_SIZE - 1))
        for idx in range(self.pattern_steps - 1, -1, -1):
            states[idx] = value % 3
            value //= 3
        return states

    def decode_pattern(self, pattern_id: int, velocity_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sustain = np.zeros(self.source_ticks_per_beat, dtype=np.uint8)
        onset = np.zeros(self.source_ticks_per_beat, dtype=np.uint8)
        velocity = np.zeros(self.source_ticks_per_beat, dtype=np.uint8)
        vel = np.uint8(np.clip(velocity_id, 0, 127))

        for bin_idx, state in enumerate(self.pattern_to_states(pattern_id)):
            start, end = self._bin_bounds(bin_idx)
            if state == STATE_SILENCE:
                continue
            if state == STATE_ONSET:
                sustain[start:end] = 1
                onset[start] = 1
                velocity[start:end] = vel
            elif state == STATE_SUSTAIN:
                sustain[start:end] = 1
                velocity[start:end] = vel

        return sustain, onset, velocity


def assemble_pianoroll(parsed: ParsedSequence, decoder: BeatPatternDecoder) -> Tuple[List[np.ndarray], dict, np.ndarray]:
    """Assemble parsed beats into measure-level pianorolls."""
    ticks_per_beat = decoder.source_ticks_per_beat

    measures_beats = []
    measure_time_signatures = []
    current_measure = []
    current_measure_ts = parsed.time_signature
    for beat in parsed.beats:
        if beat.is_bar_start and current_measure:
            measures_beats.append(current_measure)
            measure_time_signatures.append(current_measure_ts)
            current_measure = []
        if beat.is_bar_start and beat.time_signature is not None:
            current_measure_ts = beat.time_signature
        current_measure.append(beat)
    if current_measure:
        measures_beats.append(current_measure)
        measure_time_signatures.append(current_measure_ts)

    measures = []
    for beats in measures_beats:
        pr = np.zeros((6, 88, len(beats) * ticks_per_beat), dtype=np.uint8)
        for beat_idx, beat in enumerate(beats):
            t0 = beat_idx * ticks_per_beat
            t1 = t0 + ticks_per_beat
            for track_id, items in beat.tracks.items():
                ch_base = 0 if track_id == 0 else 3
                for pitch, pattern_id, velocity_id in items:
                    if pitch < 0 or pitch >= 88:
                        continue
                    sustain, onset, velocity = decoder.decode_pattern(pattern_id, velocity_id)
                    pr[ch_base, pitch, t0:t1] = sustain
                    pr[ch_base + 1, pitch, t0:t1] = onset
                    pr[ch_base + 2, pitch, t0:t1] = velocity
        measures.append(pr)

    metadata = {
        'time_signature': parsed.time_signature,
        'bpm': parsed.bpm,
        'num_measures': len(measures),
        'ticks_per_beat': ticks_per_beat,
        'num_channels': 6,
        'has_velocity': True,
        'has_pedal': False,
        'is_first': True,
        'is_last': True,
    }

    measure_info_rows = []
    for ts in measure_time_signatures:
        try:
            ts_num, ts_den = ts.split('/')
            ts_num = int(ts_num)
            ts_den = int(ts_den)
        except ValueError:
            ts_num, ts_den = 4, 4
        measure_info_rows.append([ts_num, ts_den, parsed.bpm])
    measure_info = np.array(measure_info_rows, dtype=np.uint16)
    return measures, metadata, measure_info


def save_npz(measures: List[np.ndarray], metadata: dict, measure_info: np.ndarray, output_path: str):
    save_dict = {f'measure_{idx}': measure for idx, measure in enumerate(measures)}
    save_dict['measure_info'] = measure_info
    save_dict['metadata'] = metadata
    np.savez_compressed(output_path, **save_dict)


def _parse_time_signature(ts_value, default: tuple[int, int] = (4, 4)) -> tuple[int, int]:
    try:
        if isinstance(ts_value, str):
            num, den = ts_value.split('/')
            return int(num), int(den)
        if len(ts_value) >= 2:
            return int(ts_value[0]), int(ts_value[1])
    except Exception:
        pass
    return default


def save_midi(npz_path: str, output_dir: str):
    """Convert decoded npz pianoroll to a simple two-track MIDI file."""
    import os

    import pretty_midi

    data = np.load(npz_path, allow_pickle=True)
    metadata = data['metadata'].item()
    bpm = int(metadata.get('bpm', 120))
    ticks_per_beat = int(metadata.get('ticks_per_beat', SOURCE_TICKS_PER_BEAT))
    seconds_per_tick = 60.0 / max(bpm, 1) / max(ticks_per_beat, 1)

    measure_keys = sorted(
        [k for k in data.files if k.startswith('measure_') and k != 'measure_info'],
        key=lambda x: int(x.split('_')[1]),
    )
    if not measure_keys:
        raise ValueError(f'No measures found in {npz_path}')

    full = np.concatenate([data[k] for k in measure_keys], axis=2)
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    default_ts = _parse_time_signature(metadata.get('time_signature', '4/4'))
    if 'measure_info' in data.files:
        measure_info = data['measure_info']
        last_ts = None
        elapsed_ticks = 0
        for idx, key in enumerate(measure_keys):
            if idx < len(measure_info):
                ts_num, ts_den = _parse_time_signature(measure_info[idx], default_ts)
            else:
                ts_num, ts_den = default_ts
            current_ts = (ts_num, ts_den)
            if current_ts != last_ts:
                midi.time_signature_changes.append(
                    pretty_midi.TimeSignature(
                        numerator=ts_num,
                        denominator=ts_den,
                        time=elapsed_ticks * seconds_per_tick,
                    )
                )
                last_ts = current_ts
            elapsed_ticks += data[key].shape[2]
    else:
        ts_num, ts_den = default_ts
        midi.time_signature_changes.append(
            pretty_midi.TimeSignature(ts_num, ts_den, 0.0)
        )

    for name, ch_base in (('Melody', 0), ('Accompaniment', 3)):
        instrument = pretty_midi.Instrument(program=0, name=name)
        sustain_roll = full[ch_base] > 0
        onset_roll = full[ch_base + 1] > 0
        velocity_roll = full[ch_base + 2]

        for pitch_idx in range(sustain_roll.shape[0]):
            sustain = sustain_roll[pitch_idx]
            onset = onset_roll[pitch_idx]
            velocity = velocity_roll[pitch_idx]
            for onset_pos in np.where(onset)[0]:
                end_pos = onset_pos + 1
                while end_pos < sustain.shape[0] and sustain[end_pos]:
                    end_pos += 1

                active_vel = velocity[onset_pos:end_pos][velocity[onset_pos:end_pos] > 0]
                vel = int(np.rint(active_vel.mean())) if active_vel.size else 64
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=int(np.clip(vel, 1, 127)),
                        pitch=pitch_idx + 21,
                        start=onset_pos * seconds_per_tick,
                        end=end_pos * seconds_per_tick,
                    )
                )

        if instrument.notes:
            midi.instruments.append(instrument)

    os.makedirs(output_dir, exist_ok=True)
    midi_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(npz_path))[0] + '.mid',
    )
    midi.write(midi_path)
    return midi_path


def tokens_to_npz(tokens: List[int], output_path: str, pitch_encoding: str = PITCH_ENCODING) -> Optional[str]:
    parsed = parse_tokens(tokens, pitch_encoding=pitch_encoding)
    decoder = BeatPatternDecoder()
    measures, metadata, measure_info = assemble_pianoroll(parsed, decoder)
    if not measures:
        print("WARNING: empty sequence, nothing to generate")
        return None
    save_npz(measures, metadata, measure_info, output_path)
    return output_path


def tokens_to_midi(tokens: List[int], output_prefix: str, pitch_encoding: str = PITCH_ENCODING):
    """Decode tokens to npz and MIDI."""
    import os

    npz_path = f'{output_prefix}.npz'
    result = tokens_to_npz(tokens, npz_path, pitch_encoding=pitch_encoding)
    if result is None:
        return None

    output_dir = os.path.dirname(npz_path) or '.'
    midi_path = save_midi(npz_path, output_dir)
    print(f"wrote: {npz_path}, {midi_path}")
    return npz_path
