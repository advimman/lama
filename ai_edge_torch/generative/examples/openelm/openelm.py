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

"""Example of building an OpenELM model."""

import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="transformer.layers.{}.ffn.proj_1",
    ff_down_proj="transformer.layers.{}.ffn.proj_2",
    attn_fused_qkv_proj="transformer.layers.{}.attn.qkv_proj",
    attn_query_norm="transformer.layers.{}.attn.q_norm",
    attn_key_norm="transformer.layers.{}.attn.k_norm",
    attn_output_proj="transformer.layers.{}.attn.out_proj",
    pre_attn_norm="transformer.layers.{}.attn_norm",
    pre_ff_norm="transformer.layers.{}.ffn_norm",
    embedding="transformer.token_embeddings",
    final_norm="transformer.norm",
    lm_head=None,
)


def get_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for an OpenELM model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for an OpenELM model.
  """
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6
  )
  num_heads = [12] * 4 + [16] * 14 + [20] * 12 + [24] * 6
  num_query_groups = [3] * 4 + [4] * 14 + [5] * 12 + [6] * 6

  def make_divisible(v, d):
    """Ensures that all layers have a channel number that is divisible by d."""
    new_v = int(v + d / 2) // d * d
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
      new_v += d
    return new_v

  # The way to get intermediate size is from
  # https://huggingface.co/apple/OpenELM-3B/blob/main/modeling_openelm.py
  def get_intermediate_size(idx: int) -> int:
    return make_divisible((0.5 + 0.1 * idx) * 3072, 256)

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    return cfg.TransformerBlockConfig(
        attn_config=cfg.AttentionConfig(
            num_heads=num_heads[idx],
            head_dim=128,
            num_query_groups=num_query_groups[idx],
            rotary_base=10000,
            rotary_percentage=1.0,
            qkv_transpose_before_split=True,
            query_norm_config=norm_config,
            key_norm_config=norm_config,
        ),
        ff_config=cfg.FeedForwardConfig(
            type=cfg.FeedForwardType.SEQUENTIAL,
            activation=cfg.ActivationConfig(cfg.ActivationType.SILU_GLU),
            intermediate_size=get_intermediate_size(idx),
            pre_ff_norm_config=norm_config,
        ),
        pre_attention_norm_config=norm_config,
    )

  num_layers = 36
  config = cfg.ModelConfig(
      vocab_size=32000,
      num_layers=num_layers,
      max_seq_len=2048,
      embedding_dim=3072,
      kv_cache_max_len=kv_cache_max_len,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
  )
  return config


def get_fake_model_config(kv_cache_max_len: int = 128) -> cfg.ModelConfig:
  config = get_model_config(kv_cache_max_len)
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 2 * kv_cache_max_len
  config.embedding_dim = 128
  config.block_configs = config.block_configs[: config.num_layers]
  for block_config in config.block_configs:
    block_config.attn_config.num_heads = 3
    block_config.attn_config.head_dim = 64
    block_config.ff_config.intermediate_size = 128
  return config


def build_model(
    checkpoint_path: str, **kwargs
) -> model_builder.DecoderOnlyModel:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_model_config(**kwargs),
      tensor_names=TENSOR_NAMES,
  )
