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

"""Example of building Qwen 2.5 models."""

import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder

TENSOR_NAMES = model_builder.TENSOR_NAMES


def get_3b_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Qwen 2.5 3B model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for a SmolLM model.
  """
  attn_config = cfg.AttentionConfig(
      num_heads=16,
      head_dim=128,
      num_query_groups=2,
      rotary_base=1000000,
      rotary_percentage=1.0,
      qkv_use_bias=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=11008,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-06,
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=151936,
      num_layers=36,
      max_seq_len=32768,
      embedding_dim=2048,
      kv_cache_max_len=kv_cache_max_len,
      block_configs=block_config,
      final_norm_config=norm_config,
      enable_hlfb=True,
  )
  return config


def get_1_5b_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Qwen 2.5 1B model."""
  config = get_3b_model_config(kv_cache_max_len)
  # Qwen has only one block config.
  block_config = config.block_config(0)
  block_config.attn_config.num_heads = 12
  block_config.ff_config.intermediate_size = 8960
  config.num_layers = 28
  config.embedding_dim = 1536
  return config


def get_0_5b_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Qwen 2.5 0.5B model."""
  config = get_3b_model_config(kv_cache_max_len)
  # Qwen has only one block config.
  block_config = config.block_config(0)
  block_config.attn_config.num_heads = 14
  block_config.attn_config.head_dim = 64
  block_config.ff_config.intermediate_size = 4864
  config.num_layers = 24
  config.embedding_dim = 896
  return config


def get_fake_model_config(**kwargs) -> cfg.ModelConfig:
  config = get_3b_model_config(**kwargs)
  config.vocab_size = 128
  config.num_layers = 2
  # Qwen has only one block config.
  config.block_config(0).ff_config.intermediate_size = 64
  return config


def build_3b_model(
    checkpoint_path: str, **kwargs
) -> model_builder.DecoderOnlyModel:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_3b_model_config(**kwargs),
      tensor_names=TENSOR_NAMES,
  )


def build_1_5b_model(
    checkpoint_path: str, **kwargs
) -> model_builder.DecoderOnlyModel:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_1_5b_model_config(**kwargs),
      tensor_names=TENSOR_NAMES,
  )


def build_0_5b_model(
    checkpoint_path: str, **kwargs
) -> model_builder.DecoderOnlyModel:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_0_5b_model_config(**kwargs),
      tensor_names=TENSOR_NAMES,
  )
