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
# Model configuration class.
from dataclasses import dataclass
from dataclasses import field
import enum
from typing import Optional, Sequence, Union


@enum.unique
class ActivationType(enum.Enum):
  """Different activation functions supported by the default builder."""

  LINEAR = enum.auto()
  SILU = enum.auto()
  GELU = enum.auto()
  GELU_TANH = enum.auto()
  GELU_QUICK = enum.auto()
  GE_GLU = enum.auto()
  RELU = enum.auto()
  SILU_GLU = enum.auto()


@enum.unique
class NormalizationType(enum.Enum):
  """Different normalization functions"""

  # No normalization is applied.
  NONE = enum.auto()
  RMS_NORM = enum.auto()
  LAYER_NORM = enum.auto()
  GROUP_NORM = enum.auto()


@enum.unique
class FeedForwardType(enum.Enum):
  """Different variations of the Feed Forward module."""

  # `output = linear(act(linear(x)))`.
  SEQUENTIAL = enum.auto()
  # `output = linear_2(act(linear_1(x)) * lienar_3(x))`.
  GATED = enum.auto()


class AttentionType(enum.Enum):
  GLOBAL = enum.auto()
  LOCAL_SLIDING = enum.auto()


@dataclass
class NormalizationConfig:
  """Normalizater parameters."""

  type: NormalizationType = NormalizationType.NONE
  enable_hlfb: bool = False
  epsilon: float = 1e-5
  zero_centered: bool = False
  # Number of groups used in group normalization.
  group_num: Optional[float] = None


@dataclass
class AttentionConfig:
  """Attention model's parameters."""

  num_heads: int
  head_dim: int
  # Used to determine number of groups in grouped query attention (GQA)
  # https://arxiv.org/pdf/2305.13245.pdf
  num_query_groups: Optional[int]
  # Base of rotary positional embedding.
  rotary_base: int = 10_000
  # Percentage of Rotary Positional Embedding added Q and K projections.
  rotary_percentage: Optional[float] = None
  # Whether to transpose the query groups of qkv bundled tensor before
  # splitting into separated tensors.
  qkv_transpose_before_split: bool = False
  # Whether to use bias with Query, Key, and Value projection.
  qkv_use_bias: bool = False
  # Whether the fused q, k, v projection weights interleaves q, k, v heads.
  # If True, the projection weights are in format [q_head_0, k_head_0, v_head_0, q_head_1, k_head_1, v_head_1, ...]
  # If False, the projection weights are in format [q_head_0, q_head_1, ..., k_head_0, k_head_1, ... v_head_0, v_head_1, ...]
  qkv_fused_interleaved: bool = True
  # Whether to use bias with attention output projection.
  output_proj_use_bias: bool = False
  enable_kv_cache: bool = True
  # The normalization applied to query projection's output.
  query_norm_config: NormalizationConfig = field(
      default_factory=NormalizationConfig
  )
  # The normalization applied to key projection's output.
  key_norm_config: NormalizationConfig = field(
      default_factory=NormalizationConfig
  )
  relative_attention_num_buckets: int = 0
  relative_attention_max_distance: int = 0
  # Softcap on the output logits.
  logit_softcap: Optional[float] = None
  # The type of attention.
  attn_type: Optional[AttentionType] = None
  # The size of the sliding window used for local attention.
  sliding_window_size: Optional[int] = None


@dataclass
class ActivationConfig:
  type: ActivationType = ActivationType.LINEAR
  # Dimension of input and output, used in GeGLU.
  dim_in: Optional[int] = None
  dim_out: Optional[int] = None


@dataclass
class FeedForwardConfig:
  """FeedForward module's parameters."""

  type: FeedForwardType
  activation: ActivationConfig
  intermediate_size: int
  use_bias: bool = False
  # The normalization applied to feed forward's input.
  pre_ff_norm_config: NormalizationConfig = field(
      default_factory=NormalizationConfig
  )
  # The normalization applied to feed forward's output.
  post_ff_norm_config: NormalizationConfig = field(
      default_factory=NormalizationConfig
  )


@dataclass
class TransformerBlockConfig:
  """TransformerBlock module's parameters."""

  attn_config: AttentionConfig
  ff_config: FeedForwardConfig
  # The normalization applied to attention's input.
  pre_attention_norm_config: NormalizationConfig = field(
      default_factory=NormalizationConfig
  )
  # The normalization applied to attentions's output.
  post_attention_norm_config: NormalizationConfig = field(
      default_factory=NormalizationConfig
  )
  # If set to True, only attn_config.pre_attention_norm is applied to the input
  # and the decode's output is computed as `output = input + attn_out + ff_out`
  # where attention and feed forward are called with pre_attention_norm's
  # output.
  parallel_residual: bool = False
  # The Attention computation will include relative positional bias.
  relative_attention: bool = False


@dataclass
class ModelConfig:
  """Base configurations for building a transformer architecture."""

  vocab_size: int
  num_layers: int
  max_seq_len: int
  embedding_dim: int

  # TransformerBlockConfig for each layer block. If a single
  # TransformerBlockConfig is provided, it will be used for all layers.
  block_configs: Union[TransformerBlockConfig, Sequence[TransformerBlockConfig]]

  # The normalization applied before LM head.
  final_norm_config: NormalizationConfig = field(
      default_factory=NormalizationConfig
  )

  # Scale factor of the embedding.
  embedding_scale: Optional[float] = None

  # Use bias term within LLM's HEAD.
  lm_head_use_bias: bool = False
  # Whether LLM's HEAD shares the weight of the embedding.
  lm_head_share_weight_with_embedding: bool = True

  # Whether to turn on high-level function boundary.
  enable_hlfb: bool = False

  # The maximum sequence length of the KV cache. Should not exceed max_seq_len.
  kv_cache_max_len: int = 0

  # Default batch size of the exported model. Default value is 1.
  batch_size: int = 1

  # Softcap on the model output logits.
  final_logit_softcap: Optional[float] = None

  @property
  def kv_cache_max(self) -> int:
    if self.kv_cache_max_len > 0:
      return self.kv_cache_max_len
    return self.max_seq_len

  def block_config(self, idx: int) -> TransformerBlockConfig:
    if isinstance(self.block_configs, TransformerBlockConfig):
      return self.block_configs
    if idx < 0 or idx >= len(self.block_configs):
      raise ValueError(
          f"Index {idx} is out of range for layer configs: {self.block_configs}"
      )
    return self.block_configs[idx]
