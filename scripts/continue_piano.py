"""Piano prompt continuation: take the first N bars of a piece as prompt,
generate the rest with the trained backbone, save GT/prompt/generated MIDI.

By default the prompt comes from a **MIDI file** (converted on the fly with the
same `midi2pianonpz` pipeline used to build the training data).  Pass
`--from_dataset` to instead sample held-out test NPZs.

Usage:
  # continue from a MIDI file (or a directory of MIDIs)
  CUDA_VISIBLE_DEVICES=0 python -m scripts.continue_piano \\
      --checkpoint checkpoints/piano/backbone_best.pt \\
      --midi path/to/song.mid --prompt_bars 2

  # old behaviour: sample from the dataset test split
  CUDA_VISIBLE_DEVICES=0 python -m scripts.continue_piano \\
      --checkpoint ... --from_dataset --num_samples 5
"""

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch

from beat.vocab import VOCAB
from config import BackboneModelConfig, PianoTrainConfig
from data_prep.midi2pianonpz import MIDIToPianoNPZ, save_npz
from piano.dataset import split_files
from piano.decoder import tokens_to_midi
from piano.inference import generate, load_model
from piano.tokenizer import PianoTokenizer


def collect_midis(path: str, limit: int = 0):
    """Return a list of MIDI paths from a file or a directory."""
    if os.path.isdir(path):
        files = sorted(
            str(p) for ext in ('*.mid', '*.midi')
            for p in Path(path).glob(ext)
        )
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"--midi path not found: {path}")
    if not files:
        raise FileNotFoundError(f"no .mid/.midi files under {path}")
    return files[:limit] if limit else files


def midi_to_tokens(midi_path: str, tokenizer: PianoTokenizer):
    """MIDI file → token sequence, via the piano converter + tokenizer.

    Converts to the exact 6-channel NPZ layout the tokenizer expects (written to
    a temp file), so a MIDI prompt is encoded identically to training data.
    """
    segments, metadata, measure_info, pedal = MIDIToPianoNPZ().convert(midi_path)
    fd, tmp = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        save_npz(segments, metadata, measure_info, pedal, tmp)
        return tokenizer.encode_file(tmp)
    finally:
        os.remove(tmp)


def take_first_n_bars(tokens, n_bars: int):
    """Slice the encoded sequence to its first `n_bars` bars.

    Cut just before the (n_bars+1)-th BAR token (or return None if fewer exist).
    """
    bar_positions = [i for i, t in enumerate(tokens) if t == VOCAB.bar_token]
    if len(bar_positions) <= n_bars:
        return None
    return tokens[: bar_positions[n_bars]]


def main():
    p = argparse.ArgumentParser(
        description='Continue a piano MIDI prompt with a trained BEAT model.')
    p.add_argument('--checkpoint', required=True, help='path to a trained piano checkpoint (.pt)')
    p.add_argument('--midi', type=str, default=None,
                   help='MIDI file or directory of MIDIs to continue from (default input mode)')
    p.add_argument('--from_dataset', action='store_true',
                   help='sample prompts from the held-out test split instead of MIDI files')
    p.add_argument('--num_samples', type=int, default=5, help='number of continuations to generate')
    p.add_argument('--prompt_bars', type=int, default=2, help='number of leading bars used as the prompt')
    p.add_argument('--max_length', type=int, default=3500, help='maximum number of tokens to generate')
    p.add_argument('--temperature', type=float, default=0.85, help='sampling temperature')
    p.add_argument('--top_p', type=float, default=0.95, help='nucleus (top-p) sampling threshold')
    p.add_argument('--output_dir', type=str, default='samples/piano_continue',
                   help='directory for the prompt / generated / GT MIDI files')
    p.add_argument('--seed', type=int, default=42, help='random seed')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    print(f"seed = {args.seed}, prompt_bars = {args.prompt_bars}")

    cfg = BackboneModelConfig()
    model = load_model(args.checkpoint, cfg, device)

    tk = PianoTokenizer()

    # Build the list of (base_name, gt_tokens) prompts.
    # Default: from MIDI files; --from_dataset reverts to test-split sampling.
    samples = []   # list of (base_name, gt_tokens)
    if not args.from_dataset:
        if not args.midi:
            p.error("MIDI mode (default) needs --midi <file|dir>; "
                    "or pass --from_dataset to sample the test split.")
        midi_files = collect_midis(args.midi, limit=args.num_samples * 3)
        print(f"MIDI input: {len(midi_files)} file(s) (first: {os.path.basename(midi_files[0])})")
        for mp in midi_files:
            try:
                samples.append((Path(mp).stem, midi_to_tokens(mp, tk)))
            except Exception as e:
                print(f"  skip {os.path.basename(mp)}: convert/encode failed {type(e).__name__}: {e}")
    else:
        train_cfg = PianoTrainConfig()
        splits = split_files(
            train_cfg.data_dir,
            eval_ratio=train_cfg.eval_split_ratio,
            test_ratio=train_cfg.test_split_ratio,
            seed=train_cfg.random_seed,
        )
        candidates = splits['test'][:args.num_samples * 3]   # over-sample, skip too-short ones
        print(f"test pool: {len(candidates)} files (first: {os.path.basename(candidates[0])})")
        for path in candidates:
            try:
                samples.append((os.path.splitext(os.path.basename(path))[0], tk.encode_file(path)))
            except Exception as e:
                print(f"  skip {os.path.basename(path)}: encode failed {type(e).__name__}: {e}")

    ts = datetime.now().strftime('%m%d_%H%M')
    out_dir = os.path.join(args.output_dir, f'run_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for base, gt_tokens in samples:
        if saved >= args.num_samples:
            break

        prompt = take_first_n_bars(gt_tokens, args.prompt_bars)
        if prompt is None:
            print(f"  skip {base}: fewer than {args.prompt_bars + 1} bars")
            continue

        gen_tokens = generate(
            model, prompt, device,
            max_length=args.max_length,
            temperature=args.temperature, top_p=args.top_p,
        )

        try:
            tokens_to_midi(gt_tokens, os.path.join(out_dir, f'{base}_gt'), pitch_encoding='relative')
            tokens_to_midi(prompt, os.path.join(out_dir, f'{base}_prompt'), pitch_encoding='relative')
            tokens_to_midi(gen_tokens, os.path.join(out_dir, f'{base}_generated'), pitch_encoding='relative')
        except Exception as e:
            print(f"  ✗ {base}: MIDI write failed {type(e).__name__}: {e}")
            continue

        saved += 1
        print(f"  ✓ [{saved}/{args.num_samples}] {base}: "
              f"GT {len(gt_tokens)} tok, prompt {len(prompt)} tok, gen {len(gen_tokens)} tok")

    print(f"\ndone: {out_dir}/  ({saved}/{args.num_samples} saved)")


if __name__ == '__main__':
    main()
