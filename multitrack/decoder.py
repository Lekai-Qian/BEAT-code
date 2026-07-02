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

        # per-program note list
        per_program: Dict[int, List["MultitrackDecoder._Note"]] = {}
        beat_tick = 0
        for beat in parsed.beats:
            for program, items in beat.tracks.items():
                notes = per_program.setdefault(program, [])
                for pitch_idx, pat_id, vel_id in items:
                    vel = vel_id if vel_id > 0 else default_velocity
                    notes.extend(self._pat_to_notes(
                        pitch_idx, pat_id, vel, beat_tick, ticks_per_beat,
                    ))
            beat_tick += ticks_per_beat

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

    def _pat_to_notes(
        self,
        pitch_idx: int,
        pat_id: int,
        velocity: int,
        beat_tick: int,
        ticks_per_beat: int,
    ) -> List["MultitrackDecoder._Note"]:
        """One (pitch, PAT) triple → list of (start, end, pitch, vel) note rows.

        PAT decodes to a length-τ tri-state vector. Successive sustains chain
        onto the previous onset; silence ends the running note.
        """
        states = self.id_to_pattern(pat_id)
        rows: List["MultitrackDecoder._Note"] = []
        step_ticks = ticks_per_beat / len(states)
        cur_onset: Optional[int] = None
        for j, s in enumerate(states):
            tick = beat_tick + int(round(j * step_ticks))
            if s == self.vocab.state_onset:
                if cur_onset is not None:
                    rows.append((cur_onset, tick, pitch_idx, velocity))
                cur_onset = tick
            elif s == self.vocab.state_sustain:
                if cur_onset is None:
                    cur_onset = tick
            else:  # silence
                if cur_onset is not None:
                    rows.append((cur_onset, tick, pitch_idx, velocity))
                    cur_onset = None
        if cur_onset is not None:
            rows.append((cur_onset, beat_tick + ticks_per_beat, pitch_idx, velocity))
        return rows


# Module-level helpers --------------------------------------------------------

_TS_INV = {ts: idx for ts, idx in [
    ('2/2', 0), ('2/4', 1), ('3/4', 2), ('3/8', 3),
    ('4/4', 4), ('6/8', 5), ('9/8', 6), ('UNK', 7),
]}


def _finish_beat(result: _Parsed, beat: Optional[_Beat]):
    if beat is not None:
        result.beats.append(beat)
