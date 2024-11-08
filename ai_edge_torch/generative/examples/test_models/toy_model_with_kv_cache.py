# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A toy example which has basic transformer block (w/ externalized KV-Cache)."""

from typing import Tuple

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import torch
from torch import nn

RoPECache = Tuple[torch.Tensor, torch.Tensor]


class ToyModelWithKVCache(torch.nn.Module):

  def __init__(self, config: cfg.ModelConfig) -> None:
    super().__init__()
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
    # Toy model has only one block config.
    block_config = config.block_config(0)
    self.transformer_blocks = nn.ModuleList(
        attention.TransformerBlock(block_config, config)
        for _ in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    attn_config = block_config.attn_config
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.max_seq_len,
        dim=int(attn_config.rotary_percentage * attn_config.head_dim),
        base=attn_config.rotary_base,
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.max_seq_len,
    )
    self.config = config

  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
  ) -> Tuple[torch.Tensor, kv_utils.KVCache]:
    x = self.tok_embedding(tokens)
    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.max_seq_len]

    updated_kv_entires = []
    for i, block in enumerate(self.transformer_blocks):
      kv_entry = kv_cache.caches[i] if kv_cache else None
      x, kv_entry = block(x, (cos, sin), mask, input_pos, kv_entry)
      if kv_entry:
        updated_kv_entires.append(kv_entry)

    x = self.final_norm(x)
    updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entires))
    return {'logits': self.lm_head(x), 'kv_cache': updated_kv_cache}


def get_model_config() -> cfg.ModelConfig:
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      head_dim=4,
      num_query_groups=4,
      rotary_base=10000,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=256,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=150,
      num_layers=2,
      max_seq_len=100,
      embedding_dim=128,
      block_configs=block_config,
      final_norm_config=norm_config,
      enable_hlfb=True,
  )
  return config


def get_sample_prefill_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
  tokens = torch.unsqueeze(torch.arange(0, 100, dtype=torch.int), 0)
  input_pos = torch.arange(0, 100, dtype=torch.int)
  return tokens, input_pos


def get_sample_decode_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
  tokens = torch.tensor([[1]], dtype=torch.int)
  input_pos = torch.tensor([10])
  return tokens, input_pos
