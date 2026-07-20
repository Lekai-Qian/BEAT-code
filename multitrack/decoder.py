"""Multi-track BEAT decoder — tokens → per-track piano-roll → MIDI.

Reverses the encoding in `multitrack/tokenizer.py`. Velocity tokens are still
parsed (and respected if non-default) so a future velocity-aware multi model
trained on velocity-bearing data will Just Work without decoder changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi

from beat.codec_base import BeatTokenizerBase
from beat.vocab import VOCAB, UnifiedVocab


@dataclass
class _Beat:
    is_bar_start: bool = False
    is_rest: bool = False
    time_signature: Optional[str] = None
    # tracks[program_id_or_DRUM] = list of (pitch, pat_id, velocity)
    tracks: Dict[int, List[Tuple[int, int, int]]] = field(default_factory=dict)


@dataclass
class _Parsed:
    time_signature: str = '4/4'
    bpm: int = 120
    beats: List[_Beat] = field(default_factory=list)


_DRUM_PROGRAM = 128  # GM channel-10 sentinel; reused as program identifier in `tracks` dict


class MultitrackDecoder(BeatTokenizerBase):
    """Decode multi-track BEAT token sequences into MIDI."""

    def __init__(self, vocab: UnifiedVocab = VOCAB):
        super().__init__(vocab)

    # =========================================================================
    # Token parsing
    # =========================================================================

    def parse_tokens(self, tokens: List[int]) -> _Parsed:
        """Walk the flat token stream into a structured `_Parsed` (beats × tracks)."""
        v = self.vocab
        inv_ts = {idx: ts for ts, idx in _TS_INV.items()}
        result = _Parsed()

        current_beat: Optional[_Beat] = None
        current_track: Optional[int] = None  # program (0..127) or _DRUM_PROGRAM
        track_prev_pitch: Optional[int] = None  # for relative-pitch decode
        current_ts = result.time_signature

        i = 0
        N = len(tokens)
        while i < N:
            t = tokens[i]

            # specials we skip outright
            if t in (v.bos_token, v.eos_token, v.pad_token):
                i += 1; continue

            if v.is_ts(t):
                current_ts = inv_ts.get(t - v.ts_offset, '4/4')
                if current_beat is None and not result.beats:
                    result.time_signature = current_ts
                i += 1; continue

            if v.is_tem(t):
                result.bpm = v.token_to_tempo(t)
                i += 1; continue

            if t == v.bar_token:
                _finish_beat(result, current_beat)
                current_beat = _Beat(is_bar_start=True, time_signature=current_ts)
                current_track = None
                track_prev_pitch = None
                i += 1; continue

            if t == v.rest_token:
                if current_beat is None:
                    current_beat = _Beat()
                elif current_beat.tracks or current_beat.is_rest:
                    _finish_beat(result, current_beat)
                    current_beat = _Beat()
                current_beat.is_rest = True
                current_track = None
                track_prev_pitch = None
                i += 1; continue

            if t == v.beat_token:
                if current_beat is None:
                    current_beat = _Beat()
                elif not (current_beat.is_bar_start and not current_beat.tracks and not current_beat.is_rest):
                    _finish_beat(result, current_beat)
                    current_beat = _Beat()
                current_beat.is_rest = False
                current_track = None
                track_prev_pitch = None
                i += 1; continue

            # melodic track marker
            if v.is_ins(t):
                current_track = t - v.ins_offset
                track_prev_pitch = None
                if current_beat is not None:
                    current_beat.tracks.setdefault(current_track, [])
                i += 1; continue

            # drum track marker
            if t == v.ins_drum_token:
                current_track = _DRUM_PROGRAM
                track_prev_pitch = None
                if current_beat is not None:
                    current_beat.tracks.setdefault(current_track, [])
                i += 1; continue

            # melodic (PIT, PAT, VEL) triple
            if (v.is_pit(t) and i + 2 < N
                    and v.is_pat(tokens[i + 1]) and v.is_vel(tokens[i + 2])):
                d = t - v.pit_offset
                if track_prev_pitch is None:
                    pitch = d
                else:
                    pitch = track_prev_pitch - d   # paper-aligned descending
                pat_id = tokens[i + 1] - v.pat_offset
                vel = tokens[i + 2] - v.vel_offset
                if (current_beat is not None and current_track is not None
                        and 0 <= pitch < v.num_pitches):
                    current_beat.tracks.setdefault(current_track, []).append((pitch, pat_id, vel))
                    track_prev_pitch = pitch
                i += 3; continue

            # drum (DRUM_PIT, PAT, VEL) triple
            if (v.is_drum_pit(t) and i + 2 < N
                    and v.is_pat(tokens[i + 1]) and v.is_vel(tokens[i + 2])):
                pitch = t - v.drum_pit_offset
                pat_id = tokens[i + 1] - v.pat_offset
                vel = tokens[i + 2] - v.vel_offset
                if (current_beat is not None and current_track is not None
                        and 0 <= pitch < v.num_pitches):
                    current_beat.tracks.setdefault(current_track, []).append((pitch, pat_id, vel))
                i += 3; continue

            # unrecognized token — skip
            i += 1

        _finish_beat(result, current_beat)
        return result

    # =========================================================================
    # MIDI rendering
    # =========================================================================

    # one decoded note: (start_tick, end_tick, pitch_idx, velocity)
    _Note = Tuple[int, int, int, int]

    def to_midi(
        self,
        tokens: List[int],
        output_path: str,
        ticks_per_beat: int = 24,
        default_velocity: int = 100,
    ) -> str:
        """Decode tokens → MIDI file (one PrettyMIDI Instrument per active track)."""
        parsed = self.parse_tokens(tokens)
        pm = pretty_midi.PrettyMIDI(initial_tempo=parsed.bpm)

        # Rebuild full per-program piano-rolls before scanning notes. Sustain at
        # the beginning of a beat must extend the preceding beat's onset.
        piano_rolls = self._assemble_piano_rolls(
            parsed,
            ticks_per_beat=ticks_per_beat,
            default_velocity=default_velocity,
        )
        per_program = {
            program: self._piano_roll_to_notes(roll)
            for program, roll in piano_rolls.items()
        }

        tick_seconds = (60.0 / parsed.bpm) / ticks_per_beat

        for program, notes in per_program.items():
            if program == _DRUM_PROGRAM:
                inst = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')
            else:
                inst = pretty_midi.Instrument(
                    program=program,
                    is_drum=False,
                    name=pretty_midi.program_to_instrument_name(program),
                )
            for start_tick, end_tick, pitch_idx, vel in notes:
                inst.notes.append(pretty_midi.Note(
                    velocity=int(np.clip(vel, 1, 127)),
                    pitch=21 + int(pitch_idx),  # pitch axis 0 = MIDI 21 (A0)
                    start=start_tick * tick_seconds,
                    end=max(end_tick * tick_seconds, start_tick * tick_seconds + 0.01),
                ))
            pm.instruments.append(inst)

        pm.write(output_path)
        return output_path

    def _assemble_piano_rolls(
        self,
        parsed: _Parsed,
        ticks_per_beat: int,
        default_velocity: int,
    ) -> Dict[int, np.ndarray]:
        """Assemble full `(sustain, onset, velocity)` rolls keyed by program."""
        if ticks_per_beat < 1:
            raise ValueError(f"ticks_per_beat must be positive, got {ticks_per_beat}")

        total_ticks = len(parsed.beats) * ticks_per_beat
        programs = {
            program
            for beat in parsed.beats
            for program in beat.tracks
        }
        rolls = {
            program: np.zeros((3, self.vocab.num_pitches, total_ticks), dtype=np.uint8)
            for program in programs
        }

        pattern_steps = self.vocab.pattern_steps
        for beat_idx, beat in enumerate(parsed.beats):
            beat_tick = beat_idx * ticks_per_beat
            for program, items in beat.tracks.items():
                roll = rolls[program]
                for pitch_idx, pat_id, vel_id in items:
                    if not 0 <= pitch_idx < self.vocab.num_pitches:
                        continue
                    velocity = vel_id if vel_id > 0 else default_velocity
                    states = self.id_to_pattern(pat_id)
                    for step_idx, state in enumerate(states):
                        start = beat_tick + step_idx * ticks_per_beat // pattern_steps
                        end = beat_tick + (step_idx + 1) * ticks_per_beat // pattern_steps
                        if end <= start or state == self.vocab.state_silence:
                            continue
                        roll[0, pitch_idx, start:end] = 1
                        roll[2, pitch_idx, start:end] = int(np.clip(velocity, 1, 127))
                        if state == self.vocab.state_onset:
                            roll[1, pitch_idx, start] = 1
        return rolls

    def _piano_roll_to_notes(self, roll: np.ndarray) -> List["MultitrackDecoder._Note"]:
        """Scan one full roll so active notes survive across beat boundaries."""
        sustain = roll[0] > 0
        onset = roll[1] > 0
        velocity = roll[2]
        rows: List["MultitrackDecoder._Note"] = []

        for pitch_idx in range(sustain.shape[0]):
            active_start: Optional[int] = None
            active_velocities: List[int] = []

            def finish(end_tick: int) -> None:
                nonlocal active_start, active_velocities
                if active_start is not None and end_tick > active_start:
                    vel = (
                        int(round(sum(active_velocities) / len(active_velocities)))
                        if active_velocities else self.vocab.default_vel
                    )
                    rows.append((active_start, end_tick, pitch_idx, vel))
                active_start = None
                active_velocities = []

            for tick in range(sustain.shape[1]):
                if onset[pitch_idx, tick]:
                    finish(tick)
                    active_start = tick
                elif not sustain[pitch_idx, tick]:
                    finish(tick)
                    continue
                elif active_start is None:
                    # Preserve continuation prompts that begin with sustain.
                    active_start = tick

                if sustain[pitch_idx, tick] and velocity[pitch_idx, tick] > 0:
                    active_velocities.append(int(velocity[pitch_idx, tick]))

            finish(sustain.shape[1])

        return rows


# Module-level helpers --------------------------------------------------------

_TS_INV = {ts: idx for ts, idx in [
    ('2/2', 0), ('2/4', 1), ('3/4', 2), ('3/8', 3),
    ('4/4', 4), ('6/8', 5), ('9/8', 6), ('UNK', 7),
]}


def _finish_beat(result: _Parsed, beat: Optional[_Beat]):
    if beat is not None:
        result.beats.append(beat)
