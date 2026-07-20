"""Piano inference: load backbone → generate token sequence → decode to MIDI.

Compatible with the old-vocab piano checkpoint (vocab_size=504) released before
the unified-vocab change: on load, we detect the smaller embed/lm_head matrices
and grow them to the new vocab_size (593). Existing rows stay unchanged; new
rows (DRUM_PIT / INS_DRUM, multi-only) are randomly initialized and unused by
piano inference.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.generate_piano \\
      --checkpoint checkpoints/piano/backbone_best.pt \\
      --num_samples 5
"""

from datetime import datetime
from typing import List

import torch

from beat.checkpoint import require_checkpoint_file
from beat.model import PianoLLaMA
from beat.vocab import VOCAB, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from config import BackboneModelConfig
from .decoder import tokens_to_midi


def load_model(
    checkpoint_path: str,
    model_config: BackboneModelConfig,
    device: str = 'cuda',
) -> PianoLLaMA:
    """Build the backbone and load a checkpoint, expanding vocab if needed."""
    checkpoint_path = require_checkpoint_file(checkpoint_path)
    model = PianoLLaMA(model_config)
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    ckpt_vocab = _checkpoint_vocab_size(sd)
    if ckpt_vocab is not None and ckpt_vocab < model_config.vocab_size:
        print(f"Resizing embeddings: ckpt {ckpt_vocab} → {model_config.vocab_size}; "
              f"new rows randomly initialized.")
        _expand_state_dict(sd, ckpt_vocab, model_config.vocab_size, model_config.hidden_size)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    print(f"loaded: {checkpoint_path}")
    model.to(device).eval()
    return model


def _checkpoint_vocab_size(sd: dict):
    """Infer the saved vocab size from the embed_tokens row count."""
    for key in ('model.model.embed_tokens.weight', 'model.embed_tokens.weight'):
        if key in sd:
            return sd[key].shape[0]
    return None


def _expand_state_dict(sd: dict, old_vocab: int, new_vocab: int, hidden_size: int) -> None:
    """In-place: pad embed_tokens + lm_head rows to `new_vocab` with random init.

    Mirrors what HF `model.resize_token_embeddings(new_vocab)` would do, but on
    the checkpoint dict so the model can be constructed with the new size up
    front (avoiding a second resize after load).
    """
    pad = new_vocab - old_vocab
    if pad <= 0:
        return
    for key in list(sd.keys()):
        tensor = sd[key]
        if tensor.dim() != 2:
            continue
        if tensor.shape[0] == old_vocab and tensor.shape[1] == hidden_size:
            # std follows LLaMA's default init
            extra = torch.empty(pad, hidden_size, dtype=tensor.dtype)
            torch.nn.init.normal_(extra, mean=0.0, std=0.02)
            sd[key] = torch.cat([tensor, extra], dim=0)


def make_prompt(time_signature: str = '4/4', bpm: int = 120) -> List[int]:
    """Initial prompt: [BOS, TS, TEM]."""
    return [BOS_TOKEN, VOCAB.ts_to_token(time_signature), VOCAB.tempo_to_token(bpm)]


@torch.no_grad()
def generate(
    model: PianoLLaMA,
    prompt: List[int],
    device: str = 'cuda',
    max_length: int = 3500,
    temperature: float = 0.85,
    top_p: float = 0.95,
) -> List[int]:
    """Autoregressive sampling with top-p filtering and PAD masking."""
    generated = torch.tensor([prompt], dtype=torch.long, device=device)
    past_kv = None

    for _ in range(max_length - len(prompt)):
        inp = generated[:, -1:] if past_kv else generated
        outputs = model.model(
            input_ids=inp,
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :] / temperature
        logits[0, PAD_TOKEN] = -float('inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            logits[0, sorted_indices[0][remove[0]]] = -float('inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == EOS_TOKEN:
            break

    return generated[0].cpu().tolist()
