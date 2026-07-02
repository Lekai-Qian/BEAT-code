"""PianoLLaMA backbone for BEAT-encoded piano sequences.

Thin wrapper around HuggingFace LlamaForCausalLM that lets `BackboneModelConfig`
drive both architecture and special-token IDs. Generation is implemented in
`inference.generate.generate(...)`; this class only exposes `forward` / param count.
"""

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig


def build_llama_config(model_config):
    """Build a HuggingFace LlamaConfig from BackboneModelConfig."""
    return LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        rope_theta=model_config.rope_theta,
        pad_token_id=model_config.pad_token_id,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        attention_dropout=model_config.dropout,
    )


class PianoLLaMA(nn.Module):
    """LLaMA-based BEAT generation model (piano + multi-track)."""

    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        llama_config = build_llama_config(model_config)
        self.model = LlamaForCausalLM(llama_config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False if labels is not None else True,
        )

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
