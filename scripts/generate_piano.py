"""Piano generation entry script.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.generate_piano \\
      --checkpoint checkpoints/piano/backbone_best.pt \\
      --num_samples 5
"""

import argparse
import os
from datetime import datetime

import torch

from config import BackboneModelConfig
from piano.decoder import tokens_to_midi
from piano.inference import generate, load_model, make_prompt


def main():
    p = argparse.ArgumentParser(
        description='Generate solo-piano MIDI from scratch with a trained BEAT model.')
    p.add_argument('--checkpoint', required=True, help='path to a trained piano checkpoint (.pt)')
    p.add_argument('--num_samples', type=int, default=5, help='number of pieces to generate')
    p.add_argument('--output_dir', type=str, default='samples/piano',
                   help='directory for the generated MIDI files')
    p.add_argument('--time_signature', type=str, default='4/4', help='prompt time signature, e.g. 4/4')
    p.add_argument('--bpm', type=int, default=120, help='prompt tempo in beats per minute')
    p.add_argument('--max_length', type=int, default=3500, help='maximum number of tokens to generate')
    p.add_argument('--temperature', type=float, default=1.1, help='sampling temperature')
    p.add_argument('--top_p', type=float, default=0.98, help='nucleus (top-p) sampling threshold')
    p.add_argument('--seed', type=int, default=None, help='random seed (default: random)')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = args.seed if args.seed is not None else int(torch.seed() % 2**31)
    torch.manual_seed(seed)
    print(f"seed = {seed}")

    cfg = BackboneModelConfig()
    model = load_model(args.checkpoint, cfg, device)

    ts = datetime.now().strftime('%m%d_%H%M')
    out_dir = os.path.join(args.output_dir, f'run_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    prompt = make_prompt(args.time_signature, args.bpm)
    print(f"prompt: {prompt}  → {args.num_samples} samples")

    for i in range(args.num_samples):
        tokens = generate(
            model, prompt, device,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        prefix = os.path.join(out_dir, f'sample_{i+1:03d}')
        tokens_to_midi(tokens, prefix, pitch_encoding='relative')
        print(f"  [{i+1}/{args.num_samples}] {len(tokens)} tokens → {prefix}.mid")

    print(f"\\ndone: {out_dir}/")


if __name__ == '__main__':
    main()
