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

"""Example of building a Gemma1 model."""

import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_fused_qkv_proj="model.layers.{}.self_attn.qkv_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    embedding="embedder",
    final_norm="model.norm",
    lm_head=None,
)


def get_model_config_2b(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Gemma 2B model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for a Gemma 2B model.
  """
  attn_config = cfg.AttentionConfig(
      num_heads=8,
      head_dim=256,
      num_query_groups=1,
      rotary_base=10000,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=16384,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-6,
      zero_centered=True,
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=256000,
      num_layers=18,
      max_seq_len=8192,
      embedding_dim=2048,
      embedding_scale=2048**0.5,
      kv_cache_max_len=kv_cache_max_len,
      block_configs=block_config,
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      enable_hlfb=True,
  )
  return config


def get_fake_model_config(kv_cache_max_len: int = 128) -> cfg.ModelConfig:
  config = get_model_config_2b(kv_cache_max_len)
  # Gemma has only one block config.
  config.block_config(0).ff_config.intermediate_size = 128
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 2 * kv_cache_max_len
  return config


def build_2b_model(
    checkpoint_path: str, **kwargs
) -> model_builder.DecoderOnlyModel:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_model_config_2b(**kwargs),
      tensor_names=TENSOR_NAMES,
  )
