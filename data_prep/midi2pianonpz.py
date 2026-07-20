"""MIDI → piano (mel + acc) NPZ converter, 24-tick grid with per-note velocity.

This is the canonical generator for the 24-tick piano training data (the NPZ
set fed to training, before the length-split step), produced from per-piece
piano MIDI files.  XML is a *different*, older lineage (16-tick / 4-channel) and
cannot reproduce this data — use MIDI.

Output format consumed by `piano.tokenizer.PianoTokenizer`:

    measure_{i}: uint8 (6, 88, T) per measure
        channels: [mel_sus, mel_ons, mel_vel, acc_sus, acc_ons, acc_vel]
        T = num * 96 // den   (24 ticks/beat, e.g. 4/4→96, 2/4→48, 6/8→72)
        pitch axis 0 = MIDI 21 (A0), 87 = MIDI 108 (C8)
        sus/ons channels are binary {0, 1}; vel channel is MIDI velocity 1–127.
    measure_info: int32 (num_measures, 3) — per-measure (numerator, denominator, bpm).
    pedal: uint32 (num_events, 2) — global (tick_24, value) sustain-pedal (CC64)
        events, value 1 = down (CC ≥ 64), 0 = up.  Stored only; NOT applied to
        the sustain channels (durations are the raw note durations).
    metadata: pickled dict:
        time_signature  str   first-measure TS (e.g. '4/4')
        bpm             int   first tempo (rounded)
        num_measures    int
        ticks_per_beat  int   == 24
        num_channels    int   == 6
        has_velocity    bool  == True
        has_pedal       bool  == True
        is_first        bool  True for a whole-piece NPZ (split sets False)
        is_last         bool  True for a whole-piece NPZ

Track → stream mapping (reverse-engineered from the released data):
    one track: mel = the only track; acc remains empty
    multiple tracks:
        acc = the LAST instrument track (left hand / bass)
        mel = all other tracks merged (right hand / melody; voices may be split
              across several tracks in the MusicXML→MIDI export)

Measure grid:
    Bar boundaries come from `pretty_midi.get_downbeats()` (handles pickup bars
    and mid-piece time-signature changes).  A trailing bar that contains no note
    onset is dropped, matching the released data (its last bar is always voiced).
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
try:
    from tqdm import tqdm
except ImportError:                       # tqdm is optional (progress bar only)
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else iter(())

# pretty_midi warns loudly on type-0/1 quirks in these files; silence it.
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pretty_midi')


# ----- format constants (must stay in sync with piano consumer) -----
TICKS_PER_BEAT = 24
PITCH_RANGE = 88
MIN_PITCH = 21
MAX_PITCH = 108
SUSTAIN_PEDAL_CC = 64


# ----- helpers -----------------------------------------------------------------

def _ts_at(ts_changes: List[pretty_midi.TimeSignature], t: float) -> Tuple[int, int]:
    """(numerator, denominator) of the time signature in effect at time `t`."""
    num, den = 4, 4
    for ts in ts_changes:
        if ts.time <= t + 1e-6:
            num, den = ts.numerator, ts.denominator
        else:
            break
    return num, den


def _measure_width(num: int, den: int) -> int:
    """Tick width of one bar on the 24-ticks-per-beat grid."""
    return num * 96 // den


def _bpm_at(tempo_times: np.ndarray, tempi: np.ndarray, t: float) -> int:
    """Tempo (BPM, rounded) in effect at time `t`."""
    bpm = tempi[0] if len(tempi) else 120.0
    for tt, bp in zip(tempo_times, tempi):
        if tt <= t + 1e-6:
            bpm = bp
        else:
            break
    return int(round(bpm))


def _piano_track_groups(num_instruments: int) -> Tuple[List[int], List[int]]:
    """Return melody/accompaniment track indices without duplicating one-track MIDI."""
    if num_instruments < 1:
        raise ValueError("no instrument tracks")
    if num_instruments == 1:
        return [0], []
    return list(range(num_instruments - 1)), [num_instruments - 1]


# ----- main converter ----------------------------------------------------------

class MIDIToPianoNPZ:
    """Convert a piano MIDI to the 6-channel 24-tick NPZ consumed by `PianoTokenizer`."""

    def __init__(self, ticks_per_beat: int = TICKS_PER_BEAT):
        self.ticks_per_beat = ticks_per_beat

    def convert(self, midi_path: str):
        pm = pretty_midi.PrettyMIDI(midi_path)
        ppq = pm.resolution
        tpb = self.ticks_per_beat

        def to_tick(t: float) -> float:
            """MIDI time (s) → fractional position on the symbolic 24/beat grid."""
            return pm.time_to_tick(t) / ppq * tpb

        ts_changes = sorted(pm.time_signature_changes, key=lambda x: x.time)
        tempo_times, tempi = pm.get_tempo_changes()

        downbeats = list(pm.get_downbeats())
        if not downbeats:
            raise ValueError("no downbeats (empty MIDI)")
        end_time = max(pm.get_end_time(), downbeats[-1])

        starts = downbeats
        ends = starts[1:] + [end_time]
        n_measures = len(starts)
        widths = [_measure_width(*_ts_at(ts_changes, starts[k])) for k in range(n_measures)]

        instruments = pm.instruments
        if not instruments:
            raise ValueError("no instrument tracks")
        mel_idx, acc_idx = _piano_track_groups(len(instruments))

        def collect(idxs):
            out = []
            for i in idxs:
                for n in instruments[i].notes:
                    if MIN_PITCH <= n.pitch <= MAX_PITCH:
                        out.append((n.pitch, n.start, n.end, n.velocity))
            return out

        mel_notes = collect(mel_idx)
        acc_notes = collect(acc_idx)

        segments: List[np.ndarray] = []
        for k in range(n_measures):
            W = widths[k]
            s_t, e_t = starts[k], ends[k]
            base = to_tick(s_t)
            seg = np.zeros((6, PITCH_RANGE, W), dtype=np.uint8)
            for ch_sus, ch_ons, ch_vel, stream in (
                (0, 1, 2, mel_notes), (3, 4, 5, acc_notes)
            ):
                for pitch, ns, ne, vel in stream:
                    if ns >= e_t or ne <= s_t:
                        continue
                    local_start = int(round(to_tick(ns) - base))
                    local_end = int(round(to_tick(ne) - base))
                    p = pitch - MIN_PITCH
                    a = max(local_start, 0)
                    b = min(local_end, W)
                    if b <= a:
                        b = a + 1
                    if a >= W:
                        continue
                    b = min(b, W)
                    seg[ch_sus, p, a:b] = 1
                    seg[ch_vel, p, a:b] = vel
                    if 0 <= local_start < W:                 # onset only on real attack
                        seg[ch_ons, p, local_start] = 1
            segments.append(seg)

        # drop trailing all-empty bars (the released data never ends on a silent bar)
        while segments and int((segments[-1] > 0).sum()) == 0:
            segments.pop()
        n_measures = len(segments)
        if n_measures == 0:
            raise ValueError("no voiced measures")
        widths = widths[:n_measures]
        starts = starts[:n_measures]

        # per-measure (num, den, bpm)
        measure_info = np.zeros((n_measures, 3), dtype=np.int32)
        for k in range(n_measures):
            num, den = _ts_at(ts_changes, starts[k])
            measure_info[k] = (num, den, _bpm_at(tempo_times, tempi, starts[k]))

        # global sustain-pedal (CC64) events on the 24-tick grid
        total_ticks = int(sum(widths))
        pedal_events = []
        for inst in instruments:
            for cc in inst.control_changes:
                if cc.number == SUSTAIN_PEDAL_CC:
                    tick = int(round(to_tick(cc.time)))
                    if 0 <= tick <= total_ticks:
                        pedal_events.append((tick, 1 if cc.value >= 64 else 0))
        pedal_events.sort()
        pedal = (np.array(pedal_events, dtype=np.uint32)
                 if pedal_events else np.zeros((0, 2), dtype=np.uint32))

        num0, den0 = _ts_at(ts_changes, starts[0])
        metadata = {
            'time_signature': f"{num0}/{den0}",
            'bpm': _bpm_at(tempo_times, tempi, starts[0]),
            'num_measures': n_measures,
            'ticks_per_beat': self.ticks_per_beat,
            'num_channels': 6,
            'has_velocity': True,
            'has_pedal': True,
            'is_first': True,
            'is_last': True,
        }
        return segments, metadata, measure_info, pedal


# ----- save / batch ------------------------------------------------------------

def save_npz(segments: List[np.ndarray], metadata: Dict[str, Any],
             measure_info: np.ndarray, pedal: np.ndarray, output_path: str) -> None:
    save_dict: Dict[str, Any] = {
        f'measure_{i}': seg.astype(np.uint8) for i, seg in enumerate(segments)
    }
    save_dict['measure_info'] = measure_info.astype(np.int32)
    save_dict['pedal'] = pedal.astype(np.uint32)
    save_dict['metadata'] = metadata
    np.savez_compressed(output_path, **save_dict)


def _process_one(args: Tuple[str, str, int, bool]) -> Tuple[str, str, str]:
    midi_path, output_dir, ticks_per_beat, overwrite = args
    out = os.path.join(output_dir, Path(midi_path).stem + '.npz')
    if not overwrite and os.path.exists(out):
        return midi_path, 'skipped', 'exists'
    try:
        conv = MIDIToPianoNPZ(ticks_per_beat=ticks_per_beat)
        segs, meta, m_info, pedal = conv.convert(midi_path)
        save_npz(segs, meta, m_info, pedal, out)
        return midi_path, 'ok', f"{meta['num_measures']} measures, TS={meta['time_signature']}"
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
        for f in tqdm(as_completed(futures), total=len(futures), desc='piano-midi'):
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
    ap = argparse.ArgumentParser(description="Convert piano MIDI to 24-tick 6-channel NPZ (one file per piece).")
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
        conv = MIDIToPianoNPZ(ticks_per_beat=args.ticks_per_beat)
        segs, meta, m_info, pedal = conv.convert(args.input)
        save_npz(segs, meta, m_info, pedal, out)
        print(f"wrote {out}")
        print(f"  {meta['num_measures']} measures, TS={meta['time_signature']} BPM={meta['bpm']}")
        print(f"  per-measure shape: {segs[0].shape} (6 × 88 × T={segs[0].shape[2]})")
        print(f"  pedal events: {len(pedal)}")
    else:
        batch_convert(args.input, args.output_dir,
                      ticks_per_beat=args.ticks_per_beat,
                      max_workers=args.max_workers,
                      overwrite=args.overwrite)


if __name__ == '__main__':
    main()
