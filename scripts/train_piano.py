"""Piano training entry — trains the unified BEAT backbone on piano data.

Usage:
  # single GPU
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_piano

  # multi-GPU (e.g. 4×)
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 \\
      -m scripts.train_piano

  # resume from existing checkpoint
  python -m scripts.train_piano --resume checkpoints/piano/backbone_best.pt

Tensorboard logs live under `<log_dir>/<run_name>/` — open with:
  tensorboard --logdir logs/piano --port 6006
"""

import argparse
import os
from datetime import datetime

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from beat.model import PianoLLaMA
from beat.vocab import PAD_TOKEN
from config import BackboneModelConfig, PianoTrainConfig
from piano.dataset import BackboneCollator, BackboneDataset
from piano.tokenizer import PianoTokenizer


def evaluate(model, loader, max_batches: int = 50) -> dict:
    """Return avg loss + perplexity over (up to) `max_batches` batches."""
    model.eval()
    total_loss, total_tokens, n_batches = 0.0, 0, 0
    for batch in loader:
        if n_batches >= max_batches:
            break
        with torch.no_grad():
            out = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
        valid = int((batch['labels'] != -100).sum().item())
        total_loss += out.loss.item() * valid
        total_tokens += valid
        n_batches += 1
    model.train()
    avg = total_loss / max(total_tokens, 1)
    ppl = min(torch.exp(torch.tensor(avg)).item(), 1e6)
    return {'loss': avg, 'perplexity': ppl, 'batches': n_batches}


def save_weights(model, accelerator, path: str) -> None:
    unwrapped = accelerator.unwrap_model(model)
    sd = unwrapped.state_dict()
    if accelerator.is_main_process:
        torch.save(sd, path)


def load_weights(model, path: str):
    if not path or not os.path.isfile(path):
        if path:
            print(f"[WARN] checkpoint not found: {path}")
        return model
    print(f"loading: {path}")
    sd = torch.load(path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing: {len(missing)} keys")
    if unexpected:
        print(f"  unexpected: {len(unexpected)} keys")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--log_dir', type=str, default=None)
    p.add_argument('--num_epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--gradient_accumulation_steps', type=int, default=None)
    p.add_argument('--run_name', type=str, default=None,
                   help='Tensorboard run subdir name (default: piano_<MMdd_HHMM>)')
    args = p.parse_args()

    model_cfg = BackboneModelConfig()
    train_cfg = PianoTrainConfig()
    for k in ('data_dir', 'output_dir', 'log_dir', 'num_epochs',
              'batch_size', 'num_workers', 'gradient_accumulation_steps'):
        v = getattr(args, k)
        if v is not None:
            setattr(train_cfg, k, v)

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        mixed_precision=train_cfg.mixed_precision,
        log_with='tensorboard',
        project_dir=train_cfg.log_dir,
    )

    # ---- data ----
    tokenizer = PianoTokenizer()
    train_ds = BackboneDataset(
        data_dir=train_cfg.data_dir, tokenizer=tokenizer,
        max_seq_len=model_cfg.train_cutoff_len, split='train',
        eval_ratio=train_cfg.eval_split_ratio,
        test_ratio=train_cfg.test_split_ratio,
        seed=train_cfg.random_seed,
    )
    eval_ds = BackboneDataset(
        data_dir=train_cfg.data_dir, tokenizer=tokenizer,
        max_seq_len=model_cfg.train_cutoff_len, split='eval',
        eval_ratio=train_cfg.eval_split_ratio,
        test_ratio=train_cfg.test_split_ratio,
        seed=train_cfg.random_seed,
    )
    collate = BackboneCollator(pad_token_id=PAD_TOKEN)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=train_cfg.num_workers,
        pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=train_cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=True,
    )

    # ---- model ----
    model = PianoLLaMA(model_cfg)
    if args.resume:
        model = load_weights(model, args.resume)
    total_p, train_p = model.count_parameters()
    if accelerator.is_main_process:
        print(f"PianoLLaMA: {total_p:,} params ({train_p:,} trainable) | vocab={model_cfg.vocab_size}")

    # ---- optimizer + scheduler ----
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                      weight_decay=train_cfg.weight_decay)
    raw_steps = len(train_loader)
    # raw_steps (before prepare) = global #micro-batches per epoch. Accelerate's
    # prepared scheduler advances num_processes times per optimizer step, so
    # num_training_steps must be the GLOBAL optimizer-step count (do NOT divide
    # by num_processes — that made cosine LR hit 0 at epoch/num_processes).
    total_steps = (raw_steps * train_cfg.num_epochs
                   // train_cfg.gradient_accumulation_steps)
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler,
    )

    steps_per_epoch = len(train_loader)
    eval_every_n_steps = max(1, int(steps_per_epoch * train_cfg.eval_every_epoch))
    save_every_n_steps = max(1, int(steps_per_epoch * train_cfg.save_every_n_epochs))

    run_name = args.run_name or f"piano_{datetime.now().strftime('%m%d_%H%M')}"
    accelerator.init_trackers(run_name)
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    if accelerator.is_main_process:
        print(f"\nstart: {train_cfg.num_epochs} epochs × {steps_per_epoch} steps/epoch")
        print(f"  eval every {eval_every_n_steps} steps | save every {save_every_n_steps} steps")
        print(f"  tensorboard: {train_cfg.log_dir}/{run_name}")

    global_step = 0
    best_loss = float('inf')
    next_save_step = save_every_n_steps

    for epoch in range(train_cfg.num_epochs):
        model.train()
        pbar = tqdm(
            enumerate(train_loader), total=steps_per_epoch,
            desc=f"epoch {epoch}", disable=not accelerator.is_main_process,
            mininterval=10,
        )
        for step_in_epoch, batch in pbar:
            with accelerator.accumulate(model):
                out = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                loss = out.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            global_step += 1

            if global_step % train_cfg.log_every_n_steps == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
                accelerator.log({
                    'train/loss': loss.item(),
                    'train/lr': lr_scheduler.get_last_lr()[0],
                    'train/epoch': epoch + step_in_epoch / steps_per_epoch,
                }, step=global_step)

            if step_in_epoch > 0 and step_in_epoch % eval_every_n_steps == 0:
                metrics = evaluate(model, eval_loader)
                if accelerator.is_main_process:
                    frac = step_in_epoch / steps_per_epoch
                    print(f"\n[eval] epoch {epoch}+{frac:.2f} | loss={metrics['loss']:.4f} | ppl={metrics['perplexity']:.1f}")
                    accelerator.log({
                        'eval/loss': metrics['loss'],
                        'eval/perplexity': metrics['perplexity'],
                    }, step=global_step)
                    if metrics['loss'] < best_loss:
                        best_loss = metrics['loss']
                        save_weights(model, accelerator,
                                     os.path.join(train_cfg.output_dir, 'backbone_best.pt'))
                        print(f"  → best (loss={best_loss:.4f})")

            if global_step >= next_save_step:
                if accelerator.is_main_process:
                    ts = datetime.now().strftime('%m%d_%H%M')
                    save_weights(
                        model, accelerator,
                        os.path.join(train_cfg.output_dir,
                                     f'backbone_step{global_step}_epoch{epoch}_{ts}.pt'),
                    )
                next_save_step += save_every_n_steps

    if accelerator.is_main_process:
        ts = datetime.now().strftime('%m%d_%H%M')
        save_weights(model, accelerator,
                     os.path.join(train_cfg.output_dir, f'backbone_final_{ts}.pt'))
        print(f"\ndone. best_loss={best_loss:.4f}")
    accelerator.end_training()


if __name__ == '__main__':
    main()
