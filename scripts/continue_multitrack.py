"""Multi-track prompt continuation: take the first N bars of a piece as prompt,
generate the rest with the trained backbone, save GT/prompt/generated MIDI.

By default the prompt comes from a **MIDI file** (converted on the fly with the
same `midi2multitracknpz` pipeline used to build the training data).  Pass
`--from_dataset` to instead sample held-out test NPZs (the old behaviour).

Usage:
  # continue from a MIDI file (or a directory of MIDIs)
  CUDA_VISIBLE_DEVICES=0 python -m scripts.continue_multitrack \\
      --checkpoint checkpoints/multitrack/backbone_best.pt \\
      --midi path/to/song.mid --prompt_bars 2

  # old behaviour: sample from the dataset test split
  CUDA_VISIBLE_DEVICES=0 python -m scripts.continue_multitrack \\
      --checkpoint checkpoints/multitrack/backbone_best.pt \\
      --from_dataset --num_samples 5
"""

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch

from beat.checkpoint import require_checkpoint_file
from beat.vocab import VOCAB, EOS_TOKEN, PAD_TOKEN
from beat.model import PianoLLaMA
from config import BackboneModelConfig, MultitrackTrainConfig
from data_prep.midi2multitracknpz import MIDIToMultitrackNPZ, save_npz
from multitrack.dataset import split_files
from multitrack.decoder import MultitrackDecoder
from multitrack.tokenizer import MultitrackTokenizer


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


def midi_to_tokens(midi_path: str, tokenizer: MultitrackTokenizer):
    """MIDI file → token sequence, via the multitrack converter + tokenizer.

    Converts to the exact NPZ layout the tokenizer expects (written to a temp
    file), so a MIDI prompt is encoded identically to training data.
    """
    segments, metadata = MIDIToMultitrackNPZ().convert(midi_path)
    fd, tmp = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        save_npz(segments, metadata, tmp)
        return tokenizer.encode_file(tmp)
    finally:
        os.remove(tmp)


def load_model(checkpoint_path: str, cfg: BackboneModelConfig, device: str) -> PianoLLaMA:
    checkpoint_path = require_checkpoint_file(checkpoint_path)
    model = PianoLLaMA(cfg)
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (first: {missing[:2]})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (first: {unexpected[:2]})")
    print(f"loaded: {checkpoint_path}")
    model.to(device).eval()
    return model


def take_first_n_bars(tokens, n_bars: int):
    """Slice the encoded sequence to contain only the first `n_bars` bars.

    Cut just before the (n_bars+1)-th BAR token (or the end if fewer bars exist).
    """
    bar_positions = [i for i, t in enumerate(tokens) if t == VOCAB.bar_token]
    if len(bar_positions) <= n_bars:
        return None
    return tokens[: bar_positions[n_bars]]


@torch.no_grad()
def generate(model, prompt, device, max_length: int, temperature: float, top_p: float):
    generated = torch.tensor([prompt], dtype=torch.long, device=device)
    past_kv = None
    for _ in range(max_length - len(prompt)):
        inp = generated[:, -1:] if past_kv else generated
        out = model.model(input_ids=inp, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :] / temperature
        logits[0, PAD_TOKEN] = -float('inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            logits[0, sorted_indices[0][remove[0]]] = -float('inf')

        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, nxt], dim=1)
        if nxt.item() == EOS_TOKEN:
            break
    return generated[0].cpu().tolist()


def main():
    p = argparse.ArgumentParser(
        description='Continue a multi-track MIDI prompt with a trained BEAT model.')
    p.add_argument('--checkpoint', required=True, help='path to a trained multi-track checkpoint (.pt)')
    p.add_argument('--midi', type=str, default=None,
                   help='MIDI file or directory of MIDIs to continue from (default input mode)')
    p.add_argument('--from_dataset', action='store_true',
                   help='sample prompts from the held-out test split instead of MIDI files')
    p.add_argument('--num_samples', type=int, default=5, help='number of continuations to generate')
    p.add_argument('--prompt_bars', type=int, default=2, help='number of leading bars used as the prompt')
    p.add_argument('--max_length', type=int, default=2048, help='maximum number of tokens to generate')
    p.add_argument('--temperature', type=float, default=1.0, help='sampling temperature')
    p.add_argument('--top_p', type=float, default=0.95, help='nucleus (top-p) sampling threshold')
    p.add_argument('--output_dir', type=str, default='samples/multitrack_continue',
                   help='directory for the prompt / generated / GT MIDI files')
    p.add_argument('--seed', type=int, default=42, help='random seed')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    print(f"seed = {args.seed}, prompt_bars = {args.prompt_bars}")

    cfg = BackboneModelConfig()
    model = load_model(args.checkpoint, cfg, device)

    tk = MultitrackTokenizer()
    dec = MultitrackDecoder()

    # Build the list of (base_name, gt_tokens) prompts.
    # Default: from MIDI files; --from_dataset reverts to the old test-split sampling.
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
        train_cfg = MultitrackTrainConfig()
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

        gt_mid     = os.path.join(out_dir, f'{base}_gt.mid')
        prompt_mid = os.path.join(out_dir, f'{base}_prompt.mid')
        gen_mid    = os.path.join(out_dir, f'{base}_generated.mid')
        try:
            dec.to_midi(gt_tokens, gt_mid)
            dec.to_midi(prompt, prompt_mid)
            dec.to_midi(gen_tokens, gen_mid)
        except Exception as e:
            print(f"  ✗ {base}: MIDI write failed {type(e).__name__}: {e}")
            continue

        saved += 1
        print(f"  ✓ [{saved}/{args.num_samples}] {base}: "
              f"GT {len(gt_tokens)} tok, prompt {len(prompt)} tok, gen {len(gen_tokens)} tok")

    print(f"\\ndone: {out_dir}/  ({saved}/{args.num_samples} saved)")


if __name__ == '__main__':
    main()
