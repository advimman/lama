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

"""Common building blocks for Attention layer."""

from typing import Optional, Tuple, Union

from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import scaled_dot_product_attention as sdpa
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
import torch
from torch import nn


def _embed_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    n_elem: int,
    rope: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Embed rotary positional embedding for query and key.

  Args:
    q (torch.Tensor): query tensor.
    k (torch.Tensor): key tensor.
    n_elem (int): number of elements to embed rotarty positional embedding.
    rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
  """
  if n_elem > 0:
    cos, sin = rope
    q_roped = rotary_pos_emb.apply_rope(
        q[..., :n_elem], cos.repeat(1, 2), sin.repeat(1, 2)
    )
    k_roped = rotary_pos_emb.apply_rope(
        k[..., :n_elem], cos.repeat(1, 2), sin.repeat(1, 2)
    )
    q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
    k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)
  return q, k


class TransformerBlock(nn.Module):

  def __init__(
      self,
      config: cfg.TransformerBlockConfig,
      model_config: cfg.ModelConfig,
  ) -> None:
    """Initialize an instance of the TransformerBlock.

    Args:
      config (cfg.TransformerBlockConfig): the configuration object for this
        transformer block.
      model_config (cfg.ModelConfig): the configuration object for the model
        this transformer block belongs to.
    """
    super().__init__()
    self.pre_atten_norm = builder.build_norm(
        model_config.embedding_dim,
        config.pre_attention_norm_config,
    )
    self.atten_func = CausalSelfAttention(
        model_config.batch_size,
        model_config.embedding_dim,
        config.attn_config,
        model_config.enable_hlfb,
    )
    self.post_atten_norm = builder.build_norm(
        model_config.embedding_dim,
        config.post_attention_norm_config,
    )
    self.ff = builder.build_ff(model_config.embedding_dim, config.ff_config)
    self.config = config

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: kv_utils.KVCacheEntry = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
    """Forward function of the TransformerBlock.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): the optional kv cache entry.

    Returns:
      output activation from this transformer block, and updated kv cache (if
      passed in).
    """
    kv = None
    if self.config.parallel_residual:
      x_norm = self.pre_atten_norm(x)
      atten_func_out = self.atten_func(x_norm, rope, mask, input_pos, kv_cache)
      if kv_cache is None:
        attn_out = atten_func_out
      else:
        attn_out, kv = atten_func_out
      ff_out = self.ff(x_norm)
      output = x + attn_out + ff_out
    else:
      x_norm = self.pre_atten_norm(x)
      atten_func_out = self.atten_func(x_norm, rope, mask, input_pos, kv_cache)
      if kv_cache is None:
        attn_out = atten_func_out
      else:
        attn_out, kv = atten_func_out
      x = x + attn_out
      x_norm = self.post_atten_norm(x)
      output = x + self.ff(x_norm)

    return output if kv is None else (output, kv)


class CausalSelfAttention(nn.Module):

  def __init__(
      self,
      batch_size: int,
      dim: int,
      config: cfg.AttentionConfig,
      enable_hlfb: bool,
  ) -> None:
    """Initialize an instance of CausalSelfAttention.

    Args:
      batch_size (int): batch size of the input tensor.
      dim (int): causal attention's input/output dimmension.
      config (cfg.AttentionConfig): attention specific configurations.
      enable_hlfb (bool): whether hlfb is enabled or not.
    """
    super().__init__()
    self.kv_cache = None
    self.batch_size = batch_size
    qkv_shape = (
        config.num_heads + 2 * config.num_query_groups
    ) * config.head_dim
    output_shape = config.num_heads * config.head_dim
    # Key, query, value projections for all heads.
    self.qkv_projection = nn.Linear(dim, qkv_shape, bias=config.qkv_use_bias)
    self.output_projection = nn.Linear(
        output_shape, dim, bias=config.output_proj_use_bias
    )
    self.query_norm = builder.build_norm(
        config.head_dim, config.query_norm_config
    )
    self.key_norm = builder.build_norm(config.head_dim, config.key_norm_config)
    self.config = config
    self.enable_hlfb = enable_hlfb
    self.sdpa_func = (
        sdpa.scaled_dot_product_attention_with_hlfb
        if enable_hlfb
        else sdpa.scaled_dot_product_attention
    )

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: Optional[kv_utils.KVCacheEntry] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
    """Forward function of the CausalSelfAttention layer, which can support

       MQA, GQA and MHA.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): The KV cache entry corresponding to this module.

    Returns:
      output activation from this self attention layer, and the updated
        KV Cach Entry (if passed in).
    """
    # Batch size, sequence length, embedding dimensionality.
    B, T, E = x.size()
    assert B == self.batch_size, (
        "batch size of input tensor must match with the batch size specified in"
        " the model configuration."
    )

    qkv = self.qkv_projection(x)

    # Assemble into a number of query groups to support MHA, MQA and GQA.
    q_per_kv = self.config.num_heads // self.config.num_query_groups
    # Each group has >=1 queries, 1 key, and 1 value.
    if self.config.qkv_transpose_before_split:
      qkv = qkv.view(B, T, -1, self.config.head_dim)
      q, k, v = qkv.split(
          (
              q_per_kv * self.config.num_query_groups,
              self.config.num_query_groups,
              self.config.num_query_groups,
          ),
          dim=-2,
      )
    else:
      qkv = qkv.view(B, T, self.config.num_query_groups, -1)
      q, k, v = qkv.split(
          (
              q_per_kv * self.config.head_dim,
              self.config.head_dim,
              self.config.head_dim,
          ),
          dim=-1,
      )

    q = self.query_norm(q)
    k = self.key_norm(k)

    q = q.reshape(B, T, -1, self.config.head_dim)
    k = k.reshape(B, T, -1, self.config.head_dim)
    v = v.reshape(B, T, -1, self.config.head_dim)

    # Compute rotary positional embedding for query and key.
    n_elem = int(self.config.rotary_percentage * self.config.head_dim)
    q, k = _embed_rope(q, k, n_elem, rope)

    if kv_cache is not None:
      kv_cache = kv_utils.update(
          kv_cache, input_pos, k, v, enable_hlfb=self.enable_hlfb
      )
      k, v = kv_cache.k_cache, kv_cache.v_cache

    y = self.sdpa_func(
        q,
        k,
        v,
        self.config.head_dim,
        mask=mask,
        softcap=self.config.logit_softcap,
    )
    y = y.reshape(B, T, -1)

    # Compute the output projection.
    y = self.output_projection(y)
    return y if kv_cache is None else (y, kv_cache)


class SelfAttention(CausalSelfAttention):
  """Non-causal Self Attention module, which is equivalent to CausalSelfAttention without mask."""

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: Optional[kv_utils.KVCacheEntry] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
    """Forward function of the SelfAttention layer, which can support MQA, GQA and MHA.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): The KV cache entry corresponding to this module.

    Returns:
      output activation from this self attention layer, and the updated
        KV Cach Entry (if passed in).
    """
    B, T, _ = x.size()
    return super().forward(
        x,
        rope=rope,
        mask=torch.zeros((B, 1, T, T), dtype=torch.float32),
        input_pos=input_pos,
    )


class CrossAttention(nn.Module):

  def __init__(
      self,
      batch_size: int,
      query_dim: int,
      cross_dim: int,
      hidden_dim: int,
      output_dim: int,
      config: cfg.AttentionConfig,
      enable_hlfb: bool,
  ):
    """Initialize an instance of CrossAttention.

    Args:
      batch_size (int): batch size of the input tensor.
      query_dim (int): query tensor's dimension.
      cross_dim (int): cross attention's dimensions, for key and value tensors.
      hidden_dim (int): hidden dimension that q, k, v tensors project to.
      output_dim (int): output tensor's dimension.
      config (cfg.AttentionConfig): attention specific configurations.
      enable_hlfb (bool): whether hlfb is enabled or not.
    """
    super().__init__()
    self.config = config
    self.n_heads = config.num_heads
    self.q_projection = nn.Linear(
        query_dim, hidden_dim, bias=config.qkv_use_bias
    )
    self.k_projection = nn.Linear(
        cross_dim, hidden_dim, bias=config.qkv_use_bias
    )
    self.v_projection = nn.Linear(
        cross_dim, hidden_dim, bias=config.qkv_use_bias
    )
    self.output_projection = nn.Linear(
        hidden_dim, output_dim, bias=config.output_proj_use_bias
    )

    self.sdpa_func = (
        sdpa.scaled_dot_product_attention_with_hlfb
        if enable_hlfb
        else sdpa.scaled_dot_product_attention
    )

  def forward(
      self,
      x: torch.Tensor,
      y: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: Optional[kv_utils.KVCacheEntry] = None,
  ):
    """Forward function of the CrossAttention layer.

    Args:
      x (torch.Tensor): the target tensor, with shape [B, target_seq_len, ...].
      y (torch.Tensor): the source tensor, with shape [B, source_seq_len, ...].
      rope (Tuple[torch.Tensor, torch.Tensor]): the optional input rope tensor.
      mask (torch.Tensor): the optional mask tensor can be broadcaseted to shape
        [B, n_heads, target_seq_len, source_seq_len].
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): The KV cache entry corresponding to this module.

    Returns:
      output activation from this cross attention layer.
    """
    batch_size = x.size()[0]
    target_seq_len = x.size()[1]
    source_seq_len = y.size()[1]

    q = self.q_projection(x)
    k = self.k_projection(y)
    v = self.v_projection(y)

    interim_shape = (batch_size, -1, self.n_heads, self.config.head_dim)
    q = q.view(interim_shape)
    k = k.view(interim_shape)
    v = v.view(interim_shape)

    # Compute rotary positional embedding for query and key.
    n_elem = int(self.config.rotary_percentage * self.config.head_dim)
    q, k = _embed_rope(q, k, n_elem, rope)

    if kv_cache is not None:
      kv_cache = kv_utils.update(
          kv_cache, input_pos, k, v, enable_hlfb=self.enable_hlfb
      )
      k, v = kv_cache.k_cache, kv_cache.v_cache
    if mask is None:
      mask = torch.zeros(
          (batch_size, 1, target_seq_len, source_seq_len), dtype=torch.float32
      )
    y = self.sdpa_func(q, k, v, self.config.head_dim, mask=mask)
    y = y.reshape(batch_size, target_seq_len, -1)

    # Compute the output projection.
    y = self.output_projection(y)
    return y if kv_cache is None else (y, kv_cache)
