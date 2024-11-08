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
# Attention modules for the T5 encoder-decoder model family.

from typing import Optional, Tuple

from ai_edge_torch.generative.layers.attention import CrossAttention
import ai_edge_torch.generative.layers.builder as builder
from ai_edge_torch.generative.layers.kv_cache import KVCache
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.layers.scaled_dot_product_attention import scaled_dot_product_attention  # NOQA
from ai_edge_torch.generative.layers.scaled_dot_product_attention import scaled_dot_product_attention_with_hlfb  # NOQA
import torch
from torch import nn

BATCH_SIZE = 1


class EncoderDecoderBlock(nn.Module):

  def __init__(
      self,
      config: cfg.TransformerBlockConfig,
      model_config: cfg.ModelConfig,
      has_relative_attention_bias: bool = False,
  ) -> None:
    """Initialize an instance of the EncoderDecoderBlock.

    Args:
      config (cfg.TransformerBlockConfig): the configuration object for this
        transformer block.
      model_config (cfg.ModelConfig): the configuration object for the model
        this transformer block belongs to.
      has_relative_attention_bias (bool): whether the self attention block has
        relative bias.
    """

    super().__init__()
    self.atten_func = T5Attention(
        BATCH_SIZE,
        model_config.embedding_dim,
        config.attn_config,
        config.pre_attention_norm_config,
        model_config.kv_cache_max,
        model_config.enable_hlfb,
        has_relative_attention_bias=has_relative_attention_bias,
    )
    # For a decoder, we add a cross attention.
    if model_config.is_decoder:
      self.cross_atten_func = T5Attention(
          BATCH_SIZE,
          model_config.embedding_dim,
          config.attn_config,
          config.pre_attention_norm_config,
          model_config.kv_cache_max,
          model_config.enable_hlfb,
          # Cross Attention does not have relative attention bias.
          has_relative_attention_bias=False,
      )
    else:
      self.cross_atten_func = None

    self.post_atten_norm = builder.build_norm(
        model_config.embedding_dim,
        config.post_attention_norm_config,
    )
    self.ff = builder.build_ff(model_config.embedding_dim, config.ff_config)
    self.config = config

  def forward(
      self,
      x: torch.Tensor,
      input_pos: Optional[torch.Tensor] = None,
      mask: Optional[torch.Tensor] = None,
      relative_position: Optional[torch.Tensor] = None,
      position_bias: Optional[torch.Tensor] = None,
      encoder_hidden_states: Optional[torch.Tensor] = None,
      encoder_attention_mask: Optional[torch.Tensor] = None,
      encoder_decoder_position_bias: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the EncoderDecoderBlock.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.

    Returns:
      output activation from this transformer block.
    """

    hidden_states, position_bias = self.atten_func(
        x,
        input_pos=input_pos,
        mask=mask,
        relative_position=relative_position,
        position_bias=position_bias,
    )

    attn_out = hidden_states + x

    if self.cross_atten_func:
      hidden_states, encoder_decoder_position_bias = self.cross_atten_func(
          attn_out,
          input_pos=input_pos,
          key_value_states=encoder_hidden_states,
          mask=encoder_attention_mask,
          relative_position=relative_position,
          position_bias=encoder_decoder_position_bias,
      )
      attn_out = hidden_states + attn_out

    forwarded = self.post_atten_norm(attn_out)
    forwarded = self.ff(forwarded)
    hidden_states = attn_out + forwarded

    # encoder_deocder_position_bias is from CrossAttention
    return hidden_states, position_bias, encoder_decoder_position_bias


class T5Attention(CrossAttention):

  def __init__(
      self,
      batch: int,
      dim: int,
      config: cfg.AttentionConfig,
      norm_config: cfg.NormalizationConfig,
      kv_cache_max: int,
      enable_hlfb: bool,
      has_relative_attention_bias=False,
  ) -> None:
    """Initialize an instance of T5Attention.

    Args:
      dim (int): causal attention's input/output dimmension.
      config (cfg.AttentionConfig): attention specific configurations.
      norm_config (cfg.NormalizationConfig): normalization configure before
        attention.
      kv_cache_max (int): determines the size of the KV Cache buffer, if
        enabled.
      enable_hlfb (bool): whether hlfb is enabled or not.
      has_relative_attention_bias (bool): whether we compute relative bias.
    """
    super().__init__(batch, dim, dim, config, kv_cache_max, enable_hlfb)
    self.pre_atten_norm = builder.build_norm(dim, norm_config)

    self.has_relative_attention_bias = has_relative_attention_bias
    self.relative_attention_num_buckets = config.relative_attention_num_buckets
    if self.has_relative_attention_bias:
      self.relative_attention_bias = nn.Embedding(
          self.relative_attention_num_buckets, self.n_heads
      )

  def forward(
      self,
      x: torch.Tensor,
      input_pos: Optional[torch.Tensor] = None,
      key_value_states: Optional[torch.Tensor] = None,
      mask: Optional[torch.Tensor] = None,
      relative_position: Optional[torch.Tensor] = None,
      position_bias: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the T5Attention layer.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.

    Returns:
      output activation from this self attention layer.
    """

    x = self.pre_atten_norm(x)
    B, T, C = (
        x.size()
    )  # batch size, sequence length, embedding dimensionality (n_embd)
    query_states = self.q_projection(x)
    query_states = query_states.reshape(
        B, T, -1, self.config.head_dim
    )  # (B, T, nh_q, hs)

    if key_value_states is not None:
      (
          kvB,
          kvT,
          kvC,
      ) = (
          key_value_states.size()
      )  # batch size, sequence length, embedding dimensionality (n_embd)
      key_states = self.k_projection(key_value_states)
      value_states = self.v_projection(key_value_states)
      key_states = key_states.reshape(kvB, kvT, -1, self.config.head_dim)
      value_states = value_states.reshape(kvB, kvT, -1, self.config.head_dim)
    else:
      key_states = self.k_projection(x)
      value_states = self.v_projection(x)
      key_states = key_states.reshape(B, T, -1, self.config.head_dim)
      value_states = value_states.reshape(B, T, -1, self.config.head_dim)

    if key_value_states is None and self.kv_cache is not None:
      key_states, value_states = self.kv_cache.update_cache(
          input_pos, key_states, value_states
      )

    if position_bias is None:
      # handle the encoder case first
      if self.has_relative_attention_bias:
        position_bias = self.relative_attention_bias(
            relative_position
        )  # shape (query_length, key_length, num_heads)
        position_bias = position_bias.permute([0, 1, 4, 2, 3]).squeeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
      else:
        # position_bias = torch.zeros(B, self.n_heads, T, self.config.head_dim, dtype=torch.float32)
        position_bias = torch.zeros_like(mask, dtype=torch.float32)

    mask = mask + position_bias
    y = self.sdpa_func(
        query_states,
        key_states,
        value_states,
        self.config.head_dim,
        mask=mask,
        scale=1.0,
    )
    y = y.reshape(B, T, C)  # re-assemble all head outputs side by side
    # output projection
    y = self.output_projection(y)
    return y, position_bias
