"""MusicXML → multi-track NPZ converter.

Output format consumed by `multitrack.tokenizer.MultitrackTokenizer`:

    measure_{i}: uint8 (2 * num_tracks, 88, T) per measure
        channels: [t0_sus, t0_ons, t1_sus, t1_ons, ...]
        T = TICKS_PER_BEAT × beats_per_measure (TICKS_PER_BEAT = 4)
        pitch axis 0 = MIDI 21 (A0), 87 = MIDI 108 (C8)
        binary values {0, 1}; no velocity stored.
    metadata: pickled dict with keys:
        num_tracks      int           number of valid tracks (≥ 1)
        instruments     list[int]     MIDI program (0..127) per track, 128 = drums
        num_measures    int
        time_signature  str           e.g. '4/4' (first measure only)
        time_signature_idx int        index into TS_LIST (-1 if not in list)
        bpm             int|None      first tempo in the score
        tempo_text      str|None
        is_continuation bool          False for whole-piece NPZ; True if file
                                      is a middle segment (no BOS in tokens)
        total_length    int           total ticks across all measures
        resolution      int           TICKS_PER_BEAT (4)

Run as a script for batch conversion, or import `XMLToMultitrackNPZ`.
"""

from __future__ import annotations

import argparse
import os
import shutil
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

# silence music21 XML warnings
warnings.filterwarnings('ignore', category=m21.musicxml.xmlToM21.MusicXMLWarning)


# ----- format constants (must stay in sync with multitrack consumer) ---
TICKS_PER_BEAT = 4            # τ in the paper; matches consumer's pattern_steps
PITCH_RANGE = 88
MIN_PITCH = 21                # MIDI A0 → axis index 0
MAX_PITCH = 108
DRUM_PROGRAM = 128

# index of the first 5 time signatures the multi tokenizer recognises;
# any other TS gets idx = -1 (consumer falls back to 4/4).
TS_LIST = ['4/4', '3/4', '2/4', '6/8', '2/2']


# ----- music21 instrument helpers ----------------------------------------------

PERCUSSION_CLASSES = frozenset({
    'UnpitchedPercussion', 'BassDrum', 'SnareDrum', 'TomTom',
    'HiHatCymbal', 'CrashCymbals', 'RideCymbals', 'Cymbals',
    'Tambourine', 'Cowbell', 'Woodblock', 'Triangle', 'Castanets',
    'BongoDrums', 'CongaDrum', 'Agogo', 'SteelDrum', 'Cabasa',
    'Maracas', 'Vibraslap', 'Guiro', 'Claves', 'Shaker', 'Percussion',
})
PERCUSSION_KEYWORDS = ('drum', 'percussion', 'cymbal', 'snare', 'bass drum',
                       'tom', 'hi-hat', 'bongo', 'conga', 'tambourine',
                       'triangle', 'cowbell', 'drumset', 'drumkit', 'mdl')
PIANO_KEYWORDS = ('piano', 'pno', 'pianoforte', 'klavier', 'keyboard',
                  'grand piano', 'acoustic piano', 'electric piano')


def _is_percussion(part: m21.stream.Part) -> bool:
    name = (part.partName or '').lower()
    instr = part.getInstrument(returnDefault=False)
    cls = instr.__class__.__name__ if instr else None
    return cls in PERCUSSION_CLASSES or any(k in name for k in PERCUSSION_KEYWORDS)


def _program_id(part: m21.stream.Part, is_perc: bool) -> Optional[int]:
    """Return MIDI program 0-127 for melodic tracks, 128 for drums, None to skip."""
    if is_perc:
        return DRUM_PROGRAM
    instr = part.getInstrument(returnDefault=False)
    prog = instr.midiProgram if instr else None
    if prog is None:
        # piano fallback by part-name keyword
        name = (part.partName or '').lower()
        if any(k in name for k in PIANO_KEYWORDS):
            return 0
    return prog


# ----- score-level extraction --------------------------------------------------

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
    """Sequential per-measure (start_tick, duration_ticks)."""
    out, cursor = [], 0
    for idx, measure in enumerate(ref_part.getElementsByClass('Measure')):
        dur = max(1, int(round(measure.duration.quarterLength * ticks_per_beat)))
        out.append({'index': idx, 'start': cursor, 'end': cursor + dur, 'duration': dur})
        cursor += dur
    return out


# ----- per-track note extraction -----------------------------------------------

def _percussion_pitch(element) -> Optional[int]:
    if hasattr(element, 'displayStep') and hasattr(element, 'displayOctave') \
            and element.displayStep and element.displayOctave is not None:
        p = m21.pitch.Pitch(element.displayStep)
        p.octave = element.displayOctave
        return p.midi
    if hasattr(element, 'pitch') and element.pitch:
        return element.pitch.midi
    return None


def _extract_track_notes(
    part: m21.stream.Part,
    measure_boundaries: List[Dict[str, Any]],
    is_perc: bool,
    ticks_per_beat: int,
) -> List[Dict[str, Any]]:
    """Return list of dicts {pitch, abs_start, duration, tie_group, is_tie_start, is_percussion}."""
    notes: List[Dict[str, Any]] = []
    start_by_idx = {mb['index']: mb['start'] for mb in measure_boundaries}
    active_ties: Dict[int, int] = {}
    tie_group_id = 0

    for m_idx, measure in enumerate(part.getElementsByClass('Measure')):
        m_start = start_by_idx.get(m_idx)
        if m_start is None:
            continue

        for element in measure.flatten().notes:
            # 1) drum unpitched / percussion chord
            if isinstance(element, m21.percussion.PercussionChord):
                abs_start = m_start + int(round(element.offset * ticks_per_beat))
                for n in element.notes:
                    pitch = _percussion_pitch(n)
                    if pitch is not None and MIN_PITCH <= pitch <= MAX_PITCH:
                        notes.append({'pitch': pitch, 'abs_start': abs_start, 'duration': 1,
                                      'tie_group': None, 'is_tie_start': False, 'is_percussion': True})
                continue
            if isinstance(element, m21.note.Unpitched):
                pitch = _percussion_pitch(element)
                if pitch is not None and MIN_PITCH <= pitch <= MAX_PITCH:
                    abs_start = m_start + int(round(element.offset * ticks_per_beat))
                    notes.append({'pitch': pitch, 'abs_start': abs_start, 'duration': 1,
                                  'tie_group': None, 'is_tie_start': False, 'is_percussion': True})
                continue

            # 2) regular Note / Chord
            if not isinstance(element, (m21.note.Note, m21.chord.Chord)):
                continue
            abs_start = m_start + int(round(element.offset * ticks_per_beat))
            duration = max(1, int(round(element.duration.quarterLength * ticks_per_beat)))
            if isinstance(element, m21.note.Note):
                pitches = [element.pitch.midi]
                ties = [element.tie]
            else:  # Chord
                pitches = [p.midi for p in element.pitches]
                ties = [None] * len(pitches)

            for pitch, tie in zip(pitches, ties):
                if not (MIN_PITCH <= pitch <= MAX_PITCH):
                    continue
                data = {
                    'pitch': pitch,
                    'abs_start': abs_start,
                    'duration': 1 if is_perc else duration,
                    'tie_group': None,
                    'is_tie_start': False,
                    'is_percussion': is_perc,
                }
                if tie and not is_perc:
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


# ----- main converter ----------------------------------------------------------

class XMLToMultitrackNPZ:
    """Convert a MusicXML to the multi-track NPZ format consumed by `MultitrackTokenizer`."""

    def __init__(self, ticks_per_beat: int = TICKS_PER_BEAT):
        self.ticks_per_beat = ticks_per_beat

    def convert(self, xml_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        score = m21.converter.parse(xml_path)

        valid: List[Tuple[m21.stream.Part, int, bool]] = []
        for part in score.parts:
            is_perc = _is_percussion(part)
            prog = _program_id(part, is_perc)
            if prog is None:
                continue
            valid.append((part, prog, is_perc))

        if not valid:
            raise ValueError("no valid tracks (all skipped)")

        ts_obj = _first_time_signature(score)
        ts_str = ts_obj.ratioString if ts_obj is not None else '4/4'
        ts_idx = TS_LIST.index(ts_str) if ts_str in TS_LIST else -1
        bpm, tempo_text = _tempo_info(score)

        ref_part = valid[0][0]
        measure_boundaries = _measure_boundaries(ref_part, self.ticks_per_beat)
        if not measure_boundaries:
            raise ValueError("score has zero measures")
        total_length = measure_boundaries[-1]['end']

        # per-track sustain/onset roll
        num_tracks = len(valid)
        full_sus = np.zeros((num_tracks, PITCH_RANGE, total_length), dtype=np.uint8)
        full_ons = np.zeros((num_tracks, PITCH_RANGE, total_length), dtype=np.uint8)

        for t_idx, (part, prog, is_perc) in enumerate(valid):
            for note in _extract_track_notes(part, measure_boundaries, is_perc, self.ticks_per_beat):
                start = note['abs_start']
                if start >= total_length:
                    continue
                p_idx = note['pitch'] - MIN_PITCH

                if note['is_percussion']:
                    full_sus[t_idx, p_idx, start] = 1
                else:
                    end = min(start + note['duration'], total_length)
                    full_sus[t_idx, p_idx, start:end] = 1

                # onset only at note-attack (first member of a tie group)
                if note['tie_group'] is None or note['is_tie_start']:
                    full_ons[t_idx, p_idx, start] = 1

        # interleave channels into per-measure segments
        segments: List[np.ndarray] = []
        for mb in measure_boundaries:
            s, e = mb['start'], mb['end']
            channels = []
            for t_idx in range(num_tracks):
                channels.append(full_sus[t_idx, :, s:e])
                channels.append(full_ons[t_idx, :, s:e])
            segments.append(np.stack(channels, axis=0))

        metadata: Dict[str, Any] = {
            'num_tracks': num_tracks,
            'instruments': [prog for _, prog, _ in valid],
            'num_measures': len(segments),
            'time_signature': ts_str,
            'time_signature_idx': ts_idx,
            'bpm': bpm,
            'tempo_text': tempo_text,
            'is_continuation': False,
            'total_length': total_length,
            'resolution': self.ticks_per_beat,
        }
        return segments, metadata


# ----- save / batch ------------------------------------------------------------

def save_npz(segments: List[np.ndarray], metadata: Dict[str, Any], output_path: str) -> None:
    save_dict: Dict[str, Any] = {
        f'measure_{i}': seg.astype(np.uint8) for i, seg in enumerate(segments)
    }
    save_dict['metadata'] = metadata
    np.savez_compressed(output_path, **save_dict)


def _process_one(args: Tuple[str, str, int, bool]) -> Tuple[str, str, str]:
    xml_path, output_dir, ticks_per_beat, overwrite = args
    out = os.path.join(output_dir, Path(xml_path).stem + '.npz')
    if not overwrite and os.path.exists(out):
        return xml_path, 'skipped', 'exists'
    try:
        conv = XMLToMultitrackNPZ(ticks_per_beat=ticks_per_beat)
        segs, meta = conv.convert(xml_path)
        save_npz(segs, meta, out)
        return xml_path, 'ok', f"{meta['num_tracks']} tracks × {meta['num_measures']} measures"
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
        # also try .xml + .mxl
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
        for f in tqdm(as_completed(futures), total=len(futures), desc='multi'):
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
    ap = argparse.ArgumentParser(description="Convert multi-track MusicXML to (2*N, 88, T) NPZ (one file per piece).")
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
        conv = XMLToMultitrackNPZ(ticks_per_beat=args.ticks_per_beat)
        segs, meta = conv.convert(args.input)
        save_npz(segs, meta, out)
        print(f"wrote {out}")
        print(f"  {meta['num_tracks']} tracks × {meta['num_measures']} measures, "
              f"TS={meta['time_signature']} BPM={meta['bpm']}")
        print(f"  instruments: {meta['instruments']}")
    else:
        batch_convert(args.input, args.output_dir,
                      ticks_per_beat=args.ticks_per_beat,
                      max_workers=args.max_workers,
                      overwrite=args.overwrite)


if __name__ == '__main__':
    main()
