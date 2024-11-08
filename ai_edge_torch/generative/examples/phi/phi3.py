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

"""Example of building a Phi-3.5 model up to 4K tokens, not to 128K tokens."""

import math
from typing import Tuple

import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.gate_up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    attn_fused_qkv_proj="model.layers.{}.self_attn.qkv_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
    lm_head="lm_head",
)

# max_position_embeddings / original_max_position_embeddings in Phi-3.5 config.
ROPE_SCALE_FACTOR = 32

# ROPE short factor in Phi-3.5 config. According to LOPE paper and its code in
# https://github.com/microsoft/LongRoPE, these values had been searched with
# min=1.0, step-0.01 to optimize the errors of sample dataset.
ROPE_SHORT_FACTOR = [
    1.0,
    1.0199999809265137,
    1.0299999713897705,
    1.0299999713897705,
    1.0499999523162842,
    1.0499999523162842,
    1.0499999523162842,
    1.0499999523162842,
    1.0499999523162842,
    1.0699999332427979,
    1.0999999046325684,
    1.1099998950958252,
    1.1599998474121094,
    1.1599998474121094,
    1.1699998378753662,
    1.2899998426437378,
    1.339999794960022,
    1.679999828338623,
    1.7899998426437378,
    1.8199998140335083,
    1.8499997854232788,
    1.8799997568130493,
    1.9099997282028198,
    1.9399996995925903,
    1.9899996519088745,
    2.0199997425079346,
    2.0199997425079346,
    2.0199997425079346,
    2.0199997425079346,
    2.0199997425079346,
    2.0199997425079346,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0799996852874756,
    2.0899996757507324,
    2.189999580383301,
    2.2199995517730713,
    2.5899994373321533,
    2.729999542236328,
    2.749999523162842,
    2.8399994373321533,
]


def _build_rope_cache(
    size: int,
    dim: int,
    base: int,
    condense_ratio: int,
    dtype: torch.dtype,
    device: torch.device,
    theta_factors: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Precomputes Rotary Positional Embeddings for Phi-3.5 model.

  It's a modified version of attn_utils.build_rope_cache with additional
  arguments for Phi-3.5 model. It precompute Rotary Positional Embedding Sin and
  Cos values with scaling factors for quick lookup during the inference.

  Args:
      size (int): The size of the built cache.
      dim (int): Each sequence's dimmension.
      base (int, optional): Rope base value.
      condense_ratio (int, optional): The ratio by which sequence indicies are
        condensed.
      dtype (torch.dtype, optional): Output tensor's data type.
      device (torch.device, optional): Output tensor's data type.
      theta_factors (torch.Tensor, optional): A tensor of shape (dim,) used to
        scale the theta values.
      scale (float, optional): A float used to scale the rope values.

  Returns:
      Tuple[torch.Tensor, torch.Tensor]: Rope's Cosine and Sine waves.
  """
  theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
  theta = theta / theta_factors
  seq_idx = torch.arange(size) / condense_ratio
  idx_theta = torch.outer(seq_idx, theta)
  cos = torch.cos(idx_theta).to(dtype=dtype, device=device) * scale
  sin = torch.sin(idx_theta).to(dtype=dtype, device=device) * scale
  return cos, sin


class Phi3_5Mini(model_builder.DecoderOnlyModel):
  """A Phi-3.5 model built from the Edge Generative API layers."""

  def __init__(self, config: cfg.ModelConfig):
    super().__init__(config)
    attn_config = self.config.block_config(0).attn_config
    self.rope_cache = _build_rope_cache(
        size=self.config.kv_cache_max,
        dim=int(attn_config.rotary_percentage * attn_config.head_dim),
        base=attn_config.rotary_base,
        condense_ratio=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        theta_factors=torch.tensor(ROPE_SHORT_FACTOR),
        scale=math.sqrt(
            1 + math.log(ROPE_SCALE_FACTOR) / math.log(config.max_seq_len)
        ),
    )


def get_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Phi-3.5 model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for a Phi-2 model.
  """
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      head_dim=96,
      num_query_groups=32,
      rotary_base=10000,
      rotary_percentage=1.0,
      qkv_transpose_before_split=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU_GLU),
      intermediate_size=8192,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=32064,
      num_layers=32,
      max_seq_len=4096,
      kv_cache_max_len=kv_cache_max_len,
      embedding_dim=3072,
      block_configs=block_config,
      final_norm_config=norm_config,
      lm_head_share_weight_with_embedding=False,
      enable_hlfb=True,
  )
  return config


def get_fake_model_config(kv_cache_max_len: int = 128) -> cfg.ModelConfig:
  config = get_model_config(kv_cache_max_len)
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 2 * kv_cache_max_len
  # Phi-3.5 has only one block config.
  config.block_config(0).ff_config.intermediate_size = 128
  return config


def build_model(
    checkpoint_path: str, **kwargs
) -> model_builder.DecoderOnlyModel:
  """Instantiates the model instance and load checkpoint if provided."""
  config = get_model_config(**kwargs)
  model = Phi3_5Mini(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  loader.load(model)
  model.eval()
  return model
