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
# A toy example which has a single-layer transformer block.
from typing import Tuple

from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers.attention import TransformerBlock
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import torch
from torch import nn

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KV_CACHE_MAX_LEN = 100


class ToySingleLayerModel(torch.nn.Module):

  def __init__(self, config: cfg.ModelConfig) -> None:
    super().__init__()
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
    self.transformer_block = TransformerBlock(config.block_config(0), config)
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    # Toy model has only one block config.
    attn_config = config.block_config(0).attn_config
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.max_seq_len,
        dim=int(attn_config.rotary_percentage * attn_config.head_dim),
        base=attn_config.rotary_base,
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.max_seq_len,
    )
    self.config = config

  @torch.inference_mode
  def forward(self, idx: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    x = self.tok_embedding(idx)
    cos, sin = self.rope_cache

    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.max_seq_len]

    x = self.transformer_block(x, (cos, sin), mask, input_pos)
    x = self.final_norm(x)
    return self.lm_head(x)


class ToySingleLayerModelWeightSharing(torch.nn.Module):

  def __init__(self, config: cfg.ModelConfig) -> None:
    super().__init__()
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
    self.lm_head = nn.Linear(
        config.embedding_dim,
        config.vocab_size,
        bias=config.lm_head_use_bias,
    )
    self.lm_head.weight.data = self.tok_embedding.weight.data
    self.transformer_block = TransformerBlock(config.block_config(0), config)
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    # Toy model has only one block config.
    attn_config = config.block_config(0).attn_config
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.max_seq_len,
        dim=int(attn_config.rotary_percentage * attn_config.head_dim),
        base=attn_config.rotary_base,
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.max_seq_len,
    )
    self.config = config

  @torch.inference_mode
  def forward(self, idx: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    x = self.tok_embedding(idx)
    cos, sin = self.rope_cache

    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.max_seq_len]

    x = self.transformer_block(x, (cos, sin), mask, input_pos)
    x = self.final_norm(x)
    res = self.lm_head(x)
    return res


def get_model_config() -> cfg.ModelConfig:
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      head_dim=4,
      num_query_groups=4,
      rotary_base=10000,
      rotary_percentage=1.0,
      enable_kv_cache=False,
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
      vocab_size=400,
      num_layers=1,
      max_seq_len=KV_CACHE_MAX_LEN,
      embedding_dim=128,
      block_configs=block_config,
      final_norm_config=norm_config,
  )
  return config
