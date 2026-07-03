# BEAT: Tokenizing and Generating Symbolic Music by Uniform Temporal Steps

[Demo](https://lekai-qian.github.io/BEAT-ICML2026/) | [Paper](https://arxiv.org/abs/2604.19532)

Reference implementation of the ICML 2026 paper. **BEAT** (*Beat-wise Encoding
for Autoregressive Transformers*) is a single LLaMA backbone over a unified
593-token vocabulary that generates both **solo piano** and **multi-track**
symbolic music, encoding music in uniform temporal steps.

This repository contains the model, tokenizers, data-preparation pipeline, and
inference scripts. Audio examples, repetition–diversity visualisations, and
qualitative comparisons are on the [demo page](https://lekai-qian.github.io/BEAT-ICML2026/).
Released under the Apache 2.0 license.

## Structure

```text
.
├── config.py       # shared model config + per-mode train configs
├── beat/           # shared library: vocab, base-3 codec, LLaMA backbone
├── piano/          # piano mode: tokenizer / decoder / dataset / inference
├── multitrack/     # multi-track mode: tokenizer / decoder / dataset
├── data_prep/      # MIDI / MusicXML -> NPZ converters (data pipeline)
└── scripts/        # train / generate / continue entry points
```

## Software dependencies

```bash
pip install -r requirements.txt
```

The data-preparation scripts additionally need `music21` (for MusicXML) and
`pretty_midi` (for MIDI).

## Generating music

All entry points are Python modules run from the repository root. Each prints
`--help` for its full option list.

### Continue a MIDI prompt

Give the model the first few bars of a MIDI file and let it write the rest. The
input MIDI is converted on the fly with the same pipeline used to build the
training data, so any standard MIDI works.

```bash
# piano
python -m scripts.continue_piano \
    --checkpoint checkpoints/piano/backbone.pt \
    --midi song.mid --prompt_bars 2

# multi-track
python -m scripts.continue_multitrack \
    --checkpoint checkpoints/multitrack/backbone.pt \
    --midi song.mid --prompt_bars 2
```

`--midi` accepts a single file or a directory; for each prompt the script writes
`<name>_prompt.mid`, `<name>_generated.mid`, and `<name>_gt.mid` under
`--output_dir`. Pass `--from_dataset` to instead sample held-out test pieces.

### Generate from scratch

```bash
python -m scripts.generate_piano \
    --checkpoint checkpoints/piano/backbone.pt \
    --num_samples 5 --time_signature 4/4 --bpm 120
```

`generate_multitrack` takes the same options.

## Preparing your own data

The MIDI/MusicXML datasets used in the paper will be released separately
(uploaded after publication). To build the training data yourself, convert a
folder of MIDI (or MusicXML) into the NPZ format the model trains on. The
`data_prep/` modules document their NPZ formats in their module docstrings.

```bash
# piano: MIDI -> NPZ
python -m data_prep.midi2pianonpz <midi_dir> --output_dir <out_dir>

# multi-track: MusicXML -> NPZ
python -m data_prep.xml2multitracknpz <xml_dir> --output_dir <out_dir>
```

## Training

Point the model at a converted NPZ directory via the `BEAT_PIANO_DATA_DIR` /
`BEAT_MULTITRACK_DATA_DIR` environment variables (or `--data_dir`), then launch
with `accelerate`:

```bash
export BEAT_MULTITRACK_DATA_DIR=<out_dir>
accelerate launch --multi_gpu --num_processes 4 \
    -m scripts.train_multitrack
```

`train_piano` mirrors this. All hyper-parameters default from
[`config.py`](config.py) and can be overridden on the command line.

## Pretrained checkpoints

**Pretrained checkpoints will be released separately** (uploaded after
publication). Once available, place them under `checkpoints/{piano,multitrack}/`
or pass an explicit `--checkpoint`. The piano model trains against the unified
vocabulary and loads directly; the multi-track model is trained with
`scripts/train_multitrack.py`.

## Vocabulary

Size **593**, shared across modes except `INS_DRUM` / `DRUM_PIT` (multi-track
only):

| Range | Block |
| --- | --- |
| `[0, 81)` | PAT — pattern (base-3 of {silence, sustain, onset} over τ=4) |
| `[81, 209)` | VEL — MIDI velocity 0–127 |
| `[209, 297)` | PIT — relative pitch (descending) |
| `[297, 425)` | INS — GM programs 0–127 (piano: 0=melody, 1=accompaniment) |
| `425` / `426` | BEAT / BAR |
| `427` / `428` / `429` | EOS / BOS / PAD |
| `[430, 438)` | TS — time signature |
| `[438, 453)` | TEM — tempo (15 bins, 20 BPM each) |
| `503` | REST — empty-beat marker |
| `504` / `[505, 593)` | INS_DRUM / DRUM_PIT (multi-track only) |

## Citation

```bibtex
@inproceedings{beat2026,
  title     = {BEAT: Tokenizing and Generating Symbolic Music by Uniform Temporal Steps},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
