"""Regression tests for previously silent inference/tokenization failures."""

import os
import tempfile
import unittest

import numpy as np
import pretty_midi

from beat.checkpoint import require_checkpoint_file
from beat.vocab import VOCAB
from data_prep.midi2pianonpz import MIDIToPianoNPZ, _piano_track_groups
from multitrack.decoder import MultitrackDecoder
from multitrack.tokenizer import MultitrackTokenizer


class CheckpointPathTests(unittest.TestCase):
    def test_missing_checkpoint_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            require_checkpoint_file(os.path.join(tempfile.gettempdir(), "missing-beat-checkpoint.pt"))

    def test_empty_checkpoint_path_is_rejected(self):
        with self.assertRaises(ValueError):
            require_checkpoint_file("")


class MultitrackTimeSignatureTests(unittest.TestCase):
    @staticmethod
    def _encode_header(metadata):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "sample.npz")
            measure = np.zeros((2, 88, 4), dtype=np.uint8)
            payload = {
                "num_tracks": 1,
                "instruments": [0],
                "bpm": 120,
                **metadata,
            }
            np.savez_compressed(path, measure_0=measure, metadata=payload)
            return MultitrackTokenizer().encode_file(path)[:3]

    def test_time_signature_index_zero_is_preserved(self):
        header = self._encode_header({"time_signature_idx": 0})
        self.assertEqual(header[1], VOCAB.ts_to_token("4/4"))

    def test_missing_time_signature_defaults_to_four_four(self):
        header = self._encode_header({})
        self.assertEqual(header[1], VOCAB.ts_to_token("4/4"))

    def test_none_time_signature_defaults_to_four_four(self):
        header = self._encode_header({"time_signature_idx": None})
        self.assertEqual(header[1], VOCAB.ts_to_token("4/4"))

    def test_unknown_time_signature_remains_unknown(self):
        header = self._encode_header({"time_signature_idx": -1})
        self.assertEqual(header[1], VOCAB.ts_to_token("UNK"))


class MultitrackInstrumentOrderTests(unittest.TestCase):
    @staticmethod
    def _encode(instruments, pitches):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "track-order.npz")
            measure = np.zeros((2 * len(instruments), 88, 4), dtype=np.uint8)
            for track_idx, pitch in enumerate(pitches):
                measure[2 * track_idx, pitch, :] = 1
                measure[2 * track_idx + 1, pitch, 0] = 1
            metadata = {
                "num_tracks": len(instruments),
                "instruments": instruments,
                "bpm": 120,
                "time_signature_idx": 0,
            }
            np.savez_compressed(path, measure_0=measure, metadata=metadata)
            return MultitrackTokenizer().encode_file(path)

    def test_track_permutation_has_canonical_encoding(self):
        first = self._encode(instruments=[40, 0], pitches=[55, 39])
        second = self._encode(instruments=[0, 40], pitches=[39, 55])
        self.assertEqual(first, second)

    def test_tracks_are_emitted_in_program_order(self):
        tokens = self._encode(instruments=[40, 0, 128], pitches=[55, 39, 15])
        instrument_tokens = [
            token for token in tokens
            if VOCAB.is_ins(token) or token == VOCAB.ins_drum_token
        ]
        self.assertEqual(instrument_tokens, [
            VOCAB.ins_offset,
            VOCAB.ins_offset + 40,
            VOCAB.ins_drum_token,
        ])


class PianoTrackMappingTests(unittest.TestCase):
    def test_single_track_is_not_duplicated(self):
        melody, accompaniment = _piano_track_groups(1)
        self.assertEqual(melody, [0])
        self.assertEqual(accompaniment, [])

    def test_multitrack_keeps_released_data_mapping(self):
        melody, accompaniment = _piano_track_groups(3)
        self.assertEqual(melody, [0, 1])
        self.assertEqual(accompaniment, [2])

    def test_single_track_conversion_leaves_accompaniment_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            midi_path = os.path.join(tmp_dir, "single-track.mid")
            midi = pretty_midi.PrettyMIDI(initial_tempo=120)
            midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
            instrument = pretty_midi.Instrument(program=0)
            instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0, end=1))
            midi.instruments.append(instrument)
            midi.write(midi_path)

            segments, _, _, _ = MIDIToPianoNPZ().convert(midi_path)

        self.assertGreater(sum(int(segment[:3].sum()) for segment in segments), 0)
        self.assertEqual(sum(int(segment[3:].sum()) for segment in segments), 0)


class MultitrackDecoderTests(unittest.TestCase):
    @staticmethod
    def _decode_two_beat_note(second_beat_onset: bool):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "two-beat.npz")
            measure = np.zeros((2, 88, 8), dtype=np.uint8)
            measure[0, 39, :] = 1
            measure[1, 39, 0] = 1
            if second_beat_onset:
                measure[1, 39, 4] = 1
            metadata = {
                "num_tracks": 1,
                "instruments": [0],
                "bpm": 120,
                "time_signature_idx": 0,
            }
            np.savez_compressed(path, measure_0=measure, metadata=metadata)
            tokens = MultitrackTokenizer().encode_file(path)

        decoder = MultitrackDecoder()
        parsed = decoder.parse_tokens(tokens)
        rolls = decoder._assemble_piano_rolls(parsed, ticks_per_beat=24, default_velocity=100)
        return decoder._piano_roll_to_notes(rolls[0])

    def test_sustain_is_joined_across_beat_boundary(self):
        notes = self._decode_two_beat_note(second_beat_onset=False)
        self.assertEqual(notes, [(0, 48, 39, VOCAB.default_vel)])

    def test_real_onset_at_beat_boundary_retriggers_note(self):
        notes = self._decode_two_beat_note(second_beat_onset=True)
        self.assertEqual(notes, [
            (0, 24, 39, VOCAB.default_vel),
            (24, 48, 39, VOCAB.default_vel),
        ])


if __name__ == "__main__":
    unittest.main()
