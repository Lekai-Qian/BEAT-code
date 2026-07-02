"""MusicXML → piano (mel + acc) NPZ converter, with per-note velocity.

Output format consumed by `piano.tokenizer.PianoTokenizer`:

    measure_{i}: uint8 (6, 88, T) per measure
        channels: [mel_sus, mel_ons, mel_vel, acc_sus, acc_ons, acc_vel]
        T = TICKS_PER_BEAT × beats_per_measure (TICKS_PER_BEAT = 24)
        pitch axis 0 = MIDI 21 (A0), 87 = MIDI 108 (C8)
        sus/ons channels are binary {0, 1}; vel channel is MIDI velocity 0–127.
    measure_info: int32 (num_measures, 2) — per-measure (numerator, denominator)
        for mid-piece time-signature changes (tokenizer reads this optionally).
    metadata: pickled dict with keys:
        is_first, is_last     bool        True for a single whole-piece NPZ
        time_signature        str         first-measure TS (e.g. '4/4')
        bpm                   int|None    first tempo in the score
        tempo_text            str|None
        num_measures          int
        num_parts             int (== 2; mel + acc)
        num_channels          int (== 6)
        resolution            int         TICKS_PER_BEAT (24)
        total_length          int         total ticks across all measures

Notes on piano XML:
  - music21 splits a single 2-staff piano `<score-part>` into 2 Parts:
    parts[0] = treble (melody, mel), parts[1] = bass (accompaniment, acc).
    This script requires score.parts to have at least 2 entries.
  - velocity is read via `note.volume.getRealized()` (falls back to a default
    when the XML omits dynamics — gives a uniform mid-velocity track).
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import music21 as m21
import numpy as np
try:
    from tqdm import tqdm
except ImportError:                       # tqdm is optional (progress bar only)
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else iter(())

warnings.filterwarnings('ignore', category=m21.musicxml.xmlToM21.MusicXMLWarning)


# ----- format constants (must stay in sync with piano consumer) -----
TICKS_PER_BEAT = 24
PITCH_RANGE = 88
MIN_PITCH = 21
MAX_PITCH = 108
DEFAULT_VELOCITY = 64           # fallback when XML has no dynamic info

# piano subset we want to keep — others are skipped (consumer expects 4/4-like)
ALLOWED_TIME_SIGNATURES = ('2/2', '2/4', '3/4', '4/4', '6/8')

# minimum / maximum measure counts to accept (filter out very short / long pieces)
MIN_MEASURES = 16
MAX_MEASURES = 300


# ----- helpers -----------------------------------------------------------------

def _first_time_signature(score: m21.stream.Score) -> Optional[m21.meter.TimeSignature]:
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            if measure.timeSignature is not None:
                return measure.timeSignature
    return None


def _tempo_info(score: m21.stream.Score) -> Tuple[Optional[int], Optional[str]]:
    for tempo in score.flatten().getElementsByClass(m21.tempo.TempoIndication):
        bpm = getattr(tempo, 'number', None) or getattr(tempo, 'numberImplicit', None)
        text = getattr(tempo, 'text', None) or getattr(tempo, 'name', None)
        if bpm:
            return int(bpm), text
        if text:
            return None, str(text)
    return None, None


def _measure_boundaries(ref_part: m21.stream.Part, ticks_per_beat: int) -> List[Dict[str, Any]]:
    out, cursor = [], 0
    for idx, measure in enumerate(ref_part.getElementsByClass('Measure')):
        ts = measure.timeSignature
        dur = max(1, int(round(measure.duration.quarterLength * ticks_per_beat)))
        out.append({
            'index': idx,
            'start': cursor,
            'end': cursor + dur,
            'duration': dur,
            'time_signature': ts,
        })
        cursor += dur
    return out


def _measure_info_array(boundaries: List[Dict[str, Any]],
                        fallback_ts: m21.meter.TimeSignature) -> np.ndarray:
    """Per-measure (numerator, denominator) as int32 array, carrying forward last TS."""
    num = fallback_ts.numerator
    den = fallback_ts.denominator
    info = np.zeros((len(boundaries), 2), dtype=np.int32)
    for i, mb in enumerate(boundaries):
        ts = mb['time_signature']
        if ts is not None:
            num, den = ts.numerator, ts.denominator
        info[i] = (num, den)
    return info


def _get_velocity(element) -> int:
    """Return MIDI velocity (1..127) for a Note/Chord element.

    Priority:
      1. `element.volume.velocity` if set directly in the XML
      2. `element.volume.getRealized()` × 127 (uses dynamics context)
      3. `DEFAULT_VELOCITY`
    """
    try:
        vol = element.volume
        if vol is not None:
            if vol.velocity is not None:
                return int(np.clip(vol.velocity, 1, 127))
            try:
                realized = vol.getRealized()
            except Exception:
                realized = None
            if realized is not None and realized > 0:
                return int(np.clip(round(realized * 127), 1, 127))
    except Exception:
        pass
    return DEFAULT_VELOCITY


def _extract_part_notes(
    part: m21.stream.Part,
    measure_boundaries: List[Dict[str, Any]],
    ticks_per_beat: int,
) -> List[Dict[str, Any]]:
    """Return list of {pitch, abs_start, duration, tie_group, is_tie_start, velocity}."""
    notes: List[Dict[str, Any]] = []
    start_by_idx = {mb['index']: mb['start'] for mb in measure_boundaries}
    active_ties: Dict[int, int] = {}
    tie_group_id = 0

    for m_idx, measure in enumerate(part.getElementsByClass('Measure')):
        m_start = start_by_idx.get(m_idx)
        if m_start is None:
            continue

        for element in measure.flatten().notes:
            if not isinstance(element, (m21.note.Note, m21.chord.Chord)):
                continue
            abs_start = m_start + int(round(element.offset * ticks_per_beat))
            duration = max(1, int(round(element.duration.quarterLength * ticks_per_beat)))
            velocity = _get_velocity(element)

            if isinstance(element, m21.note.Note):
                pitches = [element.pitch.midi]
                ties = [element.tie]
            else:
                pitches = [p.midi for p in element.pitches]
                ties = [None] * len(pitches)

            for pitch, tie in zip(pitches, ties):
                if not (MIN_PITCH <= pitch <= MAX_PITCH):
                    continue
                data = {
                    'pitch': pitch,
                    'abs_start': abs_start,
                    'duration': duration,
                    'tie_group': None,
                    'is_tie_start': False,
                    'velocity': velocity,
                }
                if tie:
                    if tie.type == 'start':
                        tie_group_id += 1
                        active_ties[pitch] = tie_group_id
                        data['tie_group'] = tie_group_id
                        data['is_tie_start'] = True
                    elif tie.type in ('continue', 'stop'):
                        if pitch in active_ties:
                            data['tie_group'] = active_ties[pitch]
                            if tie.type == 'stop':
                                del active_ties[pitch]
                notes.append(data)
    return notes


def _notes_to_three_channel(
    notes: List[Dict[str, Any]],
    total_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-part notes into (sustain, onset, velocity) rolls."""
    sustain = np.zeros((PITCH_RANGE, total_length), dtype=np.uint8)
    onset = np.zeros((PITCH_RANGE, total_length), dtype=np.uint8)
    velocity = np.zeros((PITCH_RANGE, total_length), dtype=np.uint8)
    for note in notes:
        start = note['abs_start']
        if start >= total_length:
            continue
        p_idx = note['pitch'] - MIN_PITCH
        end = min(start + note['duration'], total_length)
        sustain[p_idx, start:end] = 1
        if note['tie_group'] is None or note['is_tie_start']:
            onset[p_idx, start] = 1
        velocity[p_idx, start:end] = note['velocity']
    return sustain, onset, velocity


# ----- main converter ----------------------------------------------------------

class XMLToPianoNPZ:
    """Convert a piano MusicXML to the 6-channel NPZ format consumed by `PianoTokenizer`."""

    def __init__(self,
                 ticks_per_beat: int = TICKS_PER_BEAT,
                 min_measures: int = MIN_MEASURES,
                 max_measures: int = MAX_MEASURES,
                 allowed_time_signatures: Tuple[str, ...] = ALLOWED_TIME_SIGNATURES):
        self.ticks_per_beat = ticks_per_beat
        self.min_measures = min_measures
        self.max_measures = max_measures
        self.allowed_ts = set(allowed_time_signatures)

    def convert(self, xml_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        score = m21.converter.parse(xml_path)
        parts = list(score.parts)
        if len(parts) < 2:
            raise ValueError(f"need ≥2 parts (mel + acc), got {len(parts)}")

        n_measures = len(parts[0].getElementsByClass('Measure'))
        if not (self.min_measures <= n_measures <= self.max_measures):
            raise ValueError(f"measure count {n_measures} outside [{self.min_measures}, {self.max_measures}]")

        ts_obj = _first_time_signature(score) or m21.meter.TimeSignature('4/4')
        if ts_obj.ratioString not in self.allowed_ts:
            raise ValueError(f"time signature {ts_obj.ratioString} not in {sorted(self.allowed_ts)}")

        bpm, tempo_text = _tempo_info(score)

        boundaries = _measure_boundaries(parts[0], self.ticks_per_beat)
        if not boundaries:
            raise ValueError("zero measures after boundary extraction")
        total_length = boundaries[-1]['end']

        mel_notes = _extract_part_notes(parts[0], boundaries, self.ticks_per_beat)
        acc_notes = _extract_part_notes(parts[1], boundaries, self.ticks_per_beat)
        mel_sus, mel_ons, mel_vel = _notes_to_three_channel(mel_notes, total_length)
        acc_sus, acc_ons, acc_vel = _notes_to_three_channel(acc_notes, total_length)

        segments: List[np.ndarray] = []
        for mb in boundaries:
            s, e = mb['start'], mb['end']
            seg = np.stack([
                mel_sus[:, s:e], mel_ons[:, s:e], mel_vel[:, s:e],
                acc_sus[:, s:e], acc_ons[:, s:e], acc_vel[:, s:e],
            ], axis=0)
            segments.append(seg)

        measure_info = _measure_info_array(boundaries, ts_obj)

        metadata: Dict[str, Any] = {
            'is_first': True,
            'is_last': True,
            'time_signature': ts_obj.ratioString,
            'bpm': bpm,
            'tempo_text': tempo_text,
            'num_measures': len(segments),
            'num_parts': 2,
            'num_channels': 6,
            'resolution': self.ticks_per_beat,
            'total_length': total_length,
        }
        return segments, metadata, measure_info


# ----- save / batch ------------------------------------------------------------

def save_npz(segments: List[np.ndarray], metadata: Dict[str, Any],
             measure_info: np.ndarray, output_path: str) -> None:
    save_dict: Dict[str, Any] = {
        f'measure_{i}': seg.astype(np.uint8) for i, seg in enumerate(segments)
    }
    save_dict['measure_info'] = measure_info.astype(np.int32)
    save_dict['metadata'] = metadata
    np.savez_compressed(output_path, **save_dict)


def _process_one(args: Tuple[str, str, int, bool]) -> Tuple[str, str, str]:
    xml_path, output_dir, ticks_per_beat, overwrite = args
    out = os.path.join(output_dir, Path(xml_path).stem + '.npz')
    if not overwrite and os.path.exists(out):
        return xml_path, 'skipped', 'exists'
    try:
        conv = XMLToPianoNPZ(ticks_per_beat=ticks_per_beat)
        segs, meta, m_info = conv.convert(xml_path)
        save_npz(segs, meta, m_info, out)
        return xml_path, 'ok', f"{meta['num_measures']} measures, TS={meta['time_signature']}"
    except Exception as e:
        return xml_path, 'error', f"{type(e).__name__}: {e}"


def batch_convert(
    input_dir: str,
    output_dir: str,
    ticks_per_beat: int = TICKS_PER_BEAT,
    max_workers: int = 8,
    overwrite: bool = False,
    pattern: str = '*.musicxml',
) -> Dict[str, List]:
    os.makedirs(output_dir, exist_ok=True)
    xml_files = sorted(Path(input_dir).glob(pattern))
    if not xml_files:
        for p in ('*.xml', '*.mxl'):
            xml_files = sorted(Path(input_dir).glob(p))
            if xml_files:
                break
    if not xml_files:
        raise ValueError(f"no XML files matching {pattern} in {input_dir}")
    print(f"found {len(xml_files)} XML files, writing NPZ to {output_dir}/")

    results = {'ok': [], 'skipped': [], 'error': []}
    args = [(str(f), output_dir, ticks_per_beat, overwrite) for f in xml_files]
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_process_one, a): a[0] for a in args}
        for f in tqdm(as_completed(futures), total=len(futures), desc='piano'):
            path, status, info = f.result()
            if status == 'error':
                results['error'].append((path, info))
            else:
                results[status].append(path)

    print(f"\n  ok      : {len(results['ok'])}")
    print(f"  skipped : {len(results['skipped'])}")
    print(f"  errors  : {len(results['error'])}")
    if results['error'][:5]:
        print('  first errors:')
        for p, e in results['error'][:5]:
            print(f"    {Path(p).name}: {e}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Convert piano MusicXML to 24-tick 6-channel NPZ (one file per piece).")
    ap.add_argument('input', help='single .musicxml file OR a directory of XMLs')
    ap.add_argument('--output_dir', required=True, help='where to write *.npz files')
    ap.add_argument('--ticks_per_beat', type=int, default=TICKS_PER_BEAT,
                    help='time resolution in ticks per beat')
    ap.add_argument('--max_workers', type=int, default=8,
                    help='number of parallel worker processes')
    ap.add_argument('--overwrite', action='store_true',
                    help='re-convert files whose NPZ already exists')
    args = ap.parse_args()

    if os.path.isfile(args.input):
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, Path(args.input).stem + '.npz')
        conv = XMLToPianoNPZ(ticks_per_beat=args.ticks_per_beat)
        segs, meta, m_info = conv.convert(args.input)
        save_npz(segs, meta, m_info, out)
        print(f"wrote {out}")
        print(f"  {meta['num_measures']} measures, TS={meta['time_signature']} BPM={meta['bpm']}")
        print(f"  per-measure shape: {segs[0].shape} (6 channels × 88 pitches × T={segs[0].shape[2]} ticks)")
    else:
        batch_convert(args.input, args.output_dir,
                      ticks_per_beat=args.ticks_per_beat,
                      max_workers=args.max_workers,
                      overwrite=args.overwrite)


if __name__ == '__main__':
    main()
