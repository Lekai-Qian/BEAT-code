"""MIDI → multi-track NPZ converter (LMD multi-track).

Source: per-piece multi-track (LMD) MIDI files.
The released multi-track training data was built from the *XML* lineage, where
music21 sometimes splits one MIDI track into several parts — so a MIDI-sourced
NPZ will not byte-match it for every file (track counts can differ).  This
converter is the clean MIDI route: one NPZ track per MIDI instrument, same
on-disk format as the consumer expects.

Output format consumed by `multitrack.tokenizer.MultitrackTokenizer`:

    measure_{i}: uint8 (2 * num_tracks, 88, T) per measure
        channels: [t0_sus, t0_ons, t1_sus, t1_ons, ...]
        T = num * 16 // den   (4 ticks/beat, e.g. 4/4→16, 2/4→8, 6/8→12)
        pitch axis 0 = MIDI 21 (A0), 87 = MIDI 108 (C8)
        binary {0, 1}; no velocity.  Drum tracks: sustain == onset (1 tick).
    metadata: pickled dict:
        num_tracks         int        number of MIDI instrument tracks
        instruments        list[int]  MIDI program 0..127 per track, 128 = drums
        num_measures       int
        time_signature     str        first-bar TS (e.g. '4/4')
        time_signature_idx int        index into TS_LIST (-1 if not listed)
        bpm                float|None  first tempo
        tempo_text         None        (MIDI carries no tempo text)
        resolution         int        == 4 (ticks per beat)
        total_length       int        total ticks across all measures
        min_pitch          int        == 21
        pitch_range        int        == 88

Track mapping: each MIDI instrument becomes one track, in file order; program is
`instrument.program` (0..127), or 128 when `instrument.is_drum`.

Measure grid: bars come from `pretty_midi.get_downbeats()` (handles pickup bars
and mid-piece TS changes; empty TS ⇒ 4/4).  Trailing all-empty bars are dropped.
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pretty_midi
try:
    from tqdm import tqdm
except ImportError:                       # tqdm is optional (progress bar only)
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else iter(())

warnings.filterwarnings('ignore', category=RuntimeWarning, module='pretty_midi')


# ----- format constants (must stay in sync with multitrack consumer) ---
TICKS_PER_BEAT = 4               # resolution = 4 ticks/beat (16th-note grid)
PITCH_RANGE = 88
MIN_PITCH = 21
MAX_PITCH = 108
DRUM_PROGRAM = 128

# the 5 time signatures the multi tokenizer recognises; others get idx = -1
TS_LIST = ['4/4', '3/4', '2/4', '6/8', '2/2']


# ----- helpers -----------------------------------------------------------------

def _ts_at(ts_changes: List[pretty_midi.TimeSignature], t: float) -> Tuple[int, int]:
    num, den = 4, 4
    for ts in ts_changes:
        if ts.time <= t + 1e-6:
            num, den = ts.numerator, ts.denominator
        else:
            break
    return num, den


def _measure_width(num: int, den: int) -> int:
    """Tick width of one bar on the 4-ticks-per-beat grid."""
    return num * 16 // den


# ----- main converter ----------------------------------------------------------

class MIDIToMultitrackNPZ:
    """Convert a multi-track MIDI to the (2·num_tracks, 88, T) NPZ format."""

    def __init__(self, ticks_per_beat: int = TICKS_PER_BEAT):
        self.ticks_per_beat = ticks_per_beat

    def convert(self, midi_path: str):
        pm = pretty_midi.PrettyMIDI(midi_path)
        ppq = pm.resolution
        tpb = self.ticks_per_beat

        def to_tick(t: float) -> float:
            return pm.time_to_tick(t) / ppq * tpb

        ts_changes = sorted(pm.time_signature_changes, key=lambda x: x.time)
        tempo_times, tempi = pm.get_tempo_changes()

        downbeats = list(pm.get_downbeats())
        if not downbeats:
            raise ValueError("no downbeats (empty MIDI)")

        instruments = pm.instruments
        if not instruments:
            raise ValueError("no instrument tracks")

        n_measures = len(downbeats)
        widths = [_measure_width(*_ts_at(ts_changes, downbeats[k])) for k in range(n_measures)]
        cum = np.cumsum([0] + widths)
        total_length = int(cum[-1])

        num_tracks = len(instruments)
        full = np.zeros((2 * num_tracks, PITCH_RANGE, total_length), dtype=np.uint8)

        instruments_prog: List[int] = []
        for t_idx, inst in enumerate(instruments):
            is_drum = inst.is_drum
            instruments_prog.append(DRUM_PROGRAM if is_drum else inst.program)
            sus_ch, ons_ch = 2 * t_idx, 2 * t_idx + 1
            for n in inst.notes:
                if not (MIN_PITCH <= n.pitch <= MAX_PITCH):
                    continue
                s = int(round(to_tick(n.start)))
                if s >= total_length:
                    continue
                p = n.pitch - MIN_PITCH
                if is_drum:
                    full[sus_ch, p, s] = 1              # drums: single-tick hit
                else:
                    e = int(round(to_tick(n.end)))
                    e = min(max(e, s + 1), total_length)
                    full[sus_ch, p, s:e] = 1
                full[ons_ch, p, s] = 1                  # onset at attack

        segments: List[np.ndarray] = [
            full[:, :, cum[k]:cum[k + 1]].copy() for k in range(n_measures)
        ]
        while segments and int((segments[-1] > 0).sum()) == 0:
            segments.pop()
        n_measures = len(segments)
        if n_measures == 0:
            raise ValueError("no voiced measures")
        total_length = int(sum(widths[:n_measures]))

        num0, den0 = _ts_at(ts_changes, downbeats[0])
        ts_str = f"{num0}/{den0}"
        bpm = float(round(tempi[0])) if len(tempi) else None

        metadata: Dict[str, Any] = {
            'num_tracks': num_tracks,
            'instruments': instruments_prog,
            'num_measures': n_measures,
            'time_signature': ts_str,
            'time_signature_idx': TS_LIST.index(ts_str) if ts_str in TS_LIST else -1,
            'bpm': bpm,
            'tempo_text': None,
            'resolution': self.ticks_per_beat,
            'total_length': total_length,
            'min_pitch': MIN_PITCH,
            'pitch_range': PITCH_RANGE,
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
    midi_path, output_dir, ticks_per_beat, overwrite = args
    out = os.path.join(output_dir, Path(midi_path).stem + '.npz')
    if not overwrite and os.path.exists(out):
        return midi_path, 'skipped', 'exists'
    try:
        conv = MIDIToMultitrackNPZ(ticks_per_beat=ticks_per_beat)
        segs, meta = conv.convert(midi_path)
        save_npz(segs, meta, out)
        return midi_path, 'ok', f"{meta['num_tracks']} tracks × {meta['num_measures']} measures"
    except Exception as e:
        return midi_path, 'error', f"{type(e).__name__}: {e}"


def batch_convert(
    input_dir: str,
    output_dir: str,
    ticks_per_beat: int = TICKS_PER_BEAT,
    max_workers: int = 8,
    overwrite: bool = False,
    pattern: str = '*.mid',
) -> Dict[str, List]:
    os.makedirs(output_dir, exist_ok=True)
    midi_files = sorted(Path(input_dir).glob(pattern))
    if not midi_files:
        midi_files = sorted(Path(input_dir).glob('*.midi'))
    if not midi_files:
        raise ValueError(f"no MIDI files matching {pattern} in {input_dir}")
    print(f"found {len(midi_files)} MIDI files, writing NPZ to {output_dir}/")

    results = {'ok': [], 'skipped': [], 'error': []}
    args = [(str(f), output_dir, ticks_per_beat, overwrite) for f in midi_files]
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_process_one, a): a[0] for a in args}
        for f in tqdm(as_completed(futures), total=len(futures), desc='multi-midi'):
            path, status, info = f.result()
            if status == 'error':
                results['error'].append((path, info))
            else:
                results[status].append(path)

    print(f"\n  ok      : {len(results['ok'])}")
    print(f"  skipped : {len(results['skipped'])}")
    print(f"  errors  : {len(results['error'])}")
    for p, e in results['error'][:5]:
        print(f"    {Path(p).name}: {e}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Convert multi-track MIDI to (2*N, 88, T) NPZ (one file per piece).")
    ap.add_argument('input', help='single .mid file OR a directory of MIDIs')
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
        conv = MIDIToMultitrackNPZ(ticks_per_beat=args.ticks_per_beat)
        segs, meta = conv.convert(args.input)
        save_npz(segs, meta, out)
        print(f"wrote {out}")
        print(f"  {meta['num_tracks']} tracks × {meta['num_measures']} measures, "
              f"TS={meta['time_signature']} BPM={meta['bpm']}")
        print(f"  instruments: {meta['instruments']}")
        print(f"  per-measure shape: {segs[0].shape}")
    else:
        batch_convert(args.input, args.output_dir,
                      ticks_per_beat=args.ticks_per_beat,
                      max_workers=args.max_workers,
                      overwrite=args.overwrite)


if __name__ == '__main__':
    main()
