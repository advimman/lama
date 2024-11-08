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
# Example of building a T5 model.

import copy
import os
from pathlib import Path
from typing import Optional

from ai_edge_torch.generative.examples.t5.t5_attention import EncoderDecoderBlock  # NOQA
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.builder as builder
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.t5_loader as loading_utils
import torch
import torch.nn as nn

ENCDEC_TENSOR_NAMES = {
    "ff_up_proj": "{prefix}.block.{}.layer.{num}.DenseReluDense.wi",
    "ff_down_proj": "{prefix}.block.{}.layer.{num}.DenseReluDense.wo",
    "attn_query_proj": "{prefix}.block.{}.layer.0.SelfAttention.q",
    "attn_key_proj": "{prefix}.block.{}.layer.0.SelfAttention.k",
    "attn_value_proj": "{prefix}.block.{}.layer.0.SelfAttention.v",
    "attn_output_proj": "{prefix}.block.{}.layer.0.SelfAttention.o",
    "relative_attn_bias": (
        "{prefix}.block.0.layer.0.SelfAttention.relative_attention_bias"
    ),
    "pre_attn_norm": "{prefix}.block.{}.layer.0.layer_norm",
    "post_attn_norm": "{prefix}.block.{}.layer.1.layer_norm",
    "final_norm": "{prefix}.final_layer_norm",
}

TENSOR_NAMES = {"lm_head": "lm_head", "embedding": "shared"}


class T5Stack(nn.Module):

  def __init__(self, config, embed_tokens=None):
    super().__init__()
    self.config = config
    self.embed_tokens = embed_tokens
    self.is_decoder = config.is_decoder
    # T5 has only one block config.
    block_config = config.block_config(0)
    self.transformer_blocks = nn.ModuleList([
        EncoderDecoderBlock(
            block_config,
            config,
            has_relative_attention_bias=bool(idx == 0),
        )
        for idx in range(config.num_layers)
    ])
    self.final_norm = builder.build_norm(
        config.embedding_dim, config.final_norm_config
    )

  def forward(
      self,
      input_ids: torch.Tensor,
      input_pos: torch.Tensor,
      attention_mask: torch.Tensor,
      relative_position: torch.Tensor,
      encoder_hidden_states: Optional[
          torch.Tensor
      ] = None,  # should be for decoder case
      encoder_attention_mask: Optional[
          torch.Tensor
      ] = None,  # should be for decoder case
  ):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds
    position_bias = None
    encoder_decoder_position_bias = None
    for _, layer_module in enumerate(self.transformer_blocks):
      # EncoderDecoderBlock.forward
      hidden_states, position_bias, encoder_decoder_position_bias = (
          layer_module(
              hidden_states,
              input_pos,
              mask=attention_mask,
              relative_position=relative_position,
              position_bias=position_bias,
              encoder_hidden_states=encoder_hidden_states,
              encoder_attention_mask=encoder_attention_mask,
              encoder_decoder_position_bias=encoder_decoder_position_bias,
          )
      )

    hidden_states = self.final_norm(hidden_states)
    return hidden_states


class T5(nn.Module):

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    self.config = config
    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )

    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    # T5 has only one block config.
    encoder_config.block_config(0).attn_config.enable_kv_cache = False
    self.encoder = T5Stack(encoder_config, self.tok_embedding)

    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    self.decoder = T5Stack(decoder_config, self.tok_embedding)
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )

    self.enc_attn_mask_cache = (
        torch.zeros(
            (config.kv_cache_max, config.kv_cache_max),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    self.dec_attn_mask_cache = attn_utils.build_causal_mask_cache(
        size=config.kv_cache_max,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # T5 has only one block config.
    attn_config = config.block_config(0).attn_config
    self.enc_rel_pos_mask = attn_utils.build_relative_position_buckets(
        bidirectional=True,
        query_length=config.kv_cache_max,
        key_length=config.kv_cache_max,
        num_buckets=attn_config.relative_attention_num_buckets,
        max_distance=attn_config.relative_attention_max_distance,
    )

    self.dec_rel_pos_mask = attn_utils.build_relative_position_buckets(
        bidirectional=False,
        query_length=config.kv_cache_max,
        key_length=config.kv_cache_max,
        num_buckets=attn_config.relative_attention_num_buckets,
        max_distance=attn_config.relative_attention_max_distance,
    )

  @torch.inference_mode
  def forward(
      self,
      input_ids: torch.Tensor,
      input_pos: torch.Tensor,
      decoder_input_ids: torch.Tensor,
      decoder_input_pos: torch.Tensor,
      pad_mask: torch.Tensor,
  ) -> torch.Tensor:
    B, T = input_ids.size()
    assert self.config.max_seq_len >= T, (
        f"Cannot forward sequence of length {T}, max seq length is only"
        f" {self.config.max_seq_len}"
    )

    enc_mask = self.enc_attn_mask_cache.index_select(2, input_pos)
    enc_mask = enc_mask[:, :, :, : self.config.kv_cache_max]
    # Mask off any "pad" tokens that shouldn't contribute to self-attention
    enc_mask[:, :, :, :] += pad_mask
    dec_mask = self.dec_attn_mask_cache.index_select(2, decoder_input_pos)
    dec_mask = dec_mask[:, :, :, : self.config.kv_cache_max]
    enc_relative_position = self.enc_rel_pos_mask.index_select(2, input_pos)
    enc_relative_position = enc_relative_position[
        :, :, :, : self.config.kv_cache_max
    ]
    dec_relative_position = self.enc_rel_pos_mask.index_select(
        2, decoder_input_pos
    )
    dec_relative_position = dec_relative_position[
        :, :, :, : self.config.kv_cache_max
    ]
    enc_attention_mask = self.enc_attn_mask_cache.index_select(
        2, decoder_input_pos
    )
    # Mask off any "pad" tokens that shouldn't contribute to cross attention
    enc_attention_mask[:, :, :, :] += pad_mask

    # Convert encoder inputs in embeddings if needed
    encoder_hidden_states = self.encoder(
        input_ids=input_ids,
        input_pos=input_pos,
        attention_mask=enc_mask,
        relative_position=enc_relative_position,
    )

    # Decode
    decoder_out = self.decoder(
        input_ids=decoder_input_ids,
        input_pos=decoder_input_pos,
        attention_mask=dec_mask,
        relative_position=dec_relative_position,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=enc_attention_mask,
    )

    # Rescale output before projecting on vocab
    # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
    sequence_output = decoder_out * (self.config.embedding_dim**-0.5)

    lm_logits = self.lm_head(sequence_output)
    return lm_logits


class T5Encoder(nn.Module):

  def __init__(self, config: cfg.ModelConfig, embedding_layer):
    super().__init__()

    self.config = config
    # Construct model layers.
    assert (
        embedding_layer != None
    ), "Passed in embedding layer should not be None!"
    self.tok_embedding = embedding_layer

    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    # T5 has only one block config.
    encoder_config.block_config(0).attn_config.enable_kv_cache = False
    self.encoder = T5Stack(encoder_config, self.tok_embedding)

    self.enc_attn_mask_cache = (
        torch.zeros(
            (config.kv_cache_max, config.kv_cache_max),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # T5 has only one block config.
    attn_config = config.block_config(0).attn_config
    self.enc_rel_pos_mask = attn_utils.build_relative_position_buckets(
        bidirectional=True,
        query_length=config.kv_cache_max,
        key_length=config.kv_cache_max,
        num_buckets=attn_config.relative_attention_num_buckets,
        max_distance=attn_config.relative_attention_max_distance,
    )

  @torch.inference_mode
  def forward(
      self,
      input_ids: torch.Tensor,
      input_pos: torch.Tensor,
      pad_mask: torch.Tensor,
  ) -> torch.Tensor:
    B, T = input_ids.size()
    assert self.config.max_seq_len >= T, (
        f"Cannot forward sequence of length {T}, max seq length is only"
        f" {self.config.max_seq_len}"
    )

    enc_mask = self.enc_attn_mask_cache.index_select(2, input_pos)
    enc_mask = enc_mask[:, :, :, : self.config.kv_cache_max]
    # Mask off any "pad" tokens that shouldn't contribute to self-attention
    enc_mask[:, :, :, :] += pad_mask
    enc_relative_position = self.enc_rel_pos_mask.index_select(2, input_pos)
    enc_relative_position = enc_relative_position[
        :, :, :, : self.config.kv_cache_max
    ]

    # Convert encoder inputs in embeddings if needed
    encoder_hidden_states = self.encoder(
        input_ids=input_ids,
        input_pos=input_pos,
        attention_mask=enc_mask,
        relative_position=enc_relative_position,
    )

    return encoder_hidden_states


class T5Decoder(nn.Module):

  def __init__(self, config: cfg.ModelConfig, embedding_layer):
    super().__init__()

    self.config = config
    # Construct model layers.
    assert (
        embedding_layer != None
    ), "Passed in embedding layer should not be None!"
    self.tok_embedding = embedding_layer

    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    self.decoder = T5Stack(decoder_config, self.tok_embedding)
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )

    self.enc_attn_mask_cache = (
        torch.zeros(
            (config.kv_cache_max, config.kv_cache_max),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # T5 has only one block config.
    attn_config = config.block_config(0).attn_config
    self.enc_rel_pos_mask = attn_utils.build_relative_position_buckets(
        bidirectional=True,
        query_length=config.kv_cache_max,
        key_length=config.kv_cache_max,
        num_buckets=attn_config.relative_attention_num_buckets,
        max_distance=attn_config.relative_attention_max_distance,
    )

    self.dec_attn_mask_cache = attn_utils.build_causal_mask_cache(
        size=config.kv_cache_max,
    )

  @torch.inference_mode
  def forward(
      self,
      encoder_hidden_states: torch.Tensor,
      decoder_input_ids: torch.Tensor,
      decoder_input_pos: torch.Tensor,
      pad_mask: torch.Tensor,
  ) -> torch.Tensor:
    dec_mask = self.dec_attn_mask_cache.index_select(2, decoder_input_pos)
    dec_mask = dec_mask[:, :, :, : self.config.kv_cache_max]
    dec_relative_position = self.enc_rel_pos_mask.index_select(
        2, decoder_input_pos
    )
    dec_relative_position = dec_relative_position[
        :, :, :, : self.config.kv_cache_max
    ]
    enc_attention_mask = self.enc_attn_mask_cache.index_select(
        2, decoder_input_pos
    )
    # Mask off any "pad" tokens that shouldn't contribute to cross attention
    enc_attention_mask[:, :, :, :] += pad_mask

    # Decode
    decoder_out = self.decoder(
        input_ids=decoder_input_ids,
        input_pos=decoder_input_pos,
        attention_mask=dec_mask,
        relative_position=dec_relative_position,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=enc_attention_mask,
    )

    # Rescale output before projecting on vocab
    # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
    sequence_output = decoder_out * (self.config.embedding_dim**-0.5)

    lm_logits = self.lm_head(sequence_output)
    return lm_logits


def get_model_config_t5() -> cfg.ModelConfig:
  attn_config = cfg.AttentionConfig(
      num_heads=12,
      head_dim=64,
      num_query_groups=12,
      qkv_use_bias=False,
      relative_attention_num_buckets=32,
      relative_attention_max_distance=128,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,
      activation=cfg.ActivationConfig(cfg.ActivationType.RELU),
      intermediate_size=3072,
  )
  # T5 Confirmed as RMS Norm and eps = 1e-6 TJA.
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-6,
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      relative_attention=True,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=32128,
      num_layers=12,
      max_seq_len=512,
      embedding_dim=768,
      block_configs=block_config,
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      enable_hlfb=True,
  )
  return config


def build_t5_model(checkpoint_path: str) -> nn.Module:
  config = get_model_config_t5()
  model = T5(config)
  # Need the encoder and decoder mappings.
  encoder_tensor_names = {
      k: v.replace("{prefix}", "encoder").replace("{num}", "1")
      for k, v in ENCDEC_TENSOR_NAMES.items()
  }
  decoder_tensor_names = ENCDEC_TENSOR_NAMES | {
      "cross_attn_query_proj": "{prefix}.block.{}.layer.1.EncDecAttention.q",
      "cross_attn_key_proj": "{prefix}.block.{}.layer.1.EncDecAttention.k",
      "cross_attn_value_proj": "{prefix}.block.{}.layer.1.EncDecAttention.v",
      "cross_attn_output_proj": "{prefix}.block.{}.layer.1.EncDecAttention.o",
      # In the decoder, the FF is layer 2 in the Transformer block
      "post_attn_norm": "{prefix}.block.{}.layer.2.layer_norm",
      # In the decoder, the cross attention is layer 1 in the Transformer block
      "pre_cross_attn_norm": "{prefix}.block.{}.layer.1.layer_norm",
  }

  decoder_tensor_names = {
      k: v.replace("{prefix}", "decoder").replace("{num}", "2")
      for k, v in decoder_tensor_names.items()
  }

  # Additional layer norms for Cross Attention in decoder
  # decoder_tensor_names["pre_attn_norm"] = "{prefix}.block.{}.layer.1.layer_norm",
  tensor_names = {
      "encoder.": loading_utils.ModelLoader.TensorNames(**encoder_tensor_names),
      "decoder.": loading_utils.ModelLoader.TensorNames(**decoder_tensor_names),
      "": loading_utils.ModelLoader.TensorNames(**TENSOR_NAMES),
  }
  loader = loading_utils.ModelLoader(checkpoint_path, names=tensor_names)
  # The embedding is shared between the encoder and decoder, so we set
  # strict=False.
  loader.load(model, strict=False, fuse_attention=False)
  return model


def build_t5_encoder_model(
    config: cfg.ModelConfig, embedding_layer, checkpoint_path: str
) -> nn.Module:
  model = T5Encoder(config, embedding_layer)
  encoder_tensor_names = {
      k: v.replace("{prefix}", "encoder").replace("{num}", "1")
      for k, v in ENCDEC_TENSOR_NAMES.items()
  }

  # Additional layer norms for Cross Attention in decoder
  # decoder_tensor_names["pre_attn_norm"] = "{prefix}.block.{}.layer.1.layer_norm",
  tensor_names = {
      "encoder.": loading_utils.ModelLoader.TensorNames(**encoder_tensor_names),
      "": loading_utils.ModelLoader.TensorNames(**TENSOR_NAMES),
  }
  loader = loading_utils.ModelLoader(checkpoint_path, names=tensor_names)
  # The embedding is shared between the encoder and decoder, so we set
  # strict=False.
  loader.load(model, strict=False, fuse_attention=False)
  return model


def build_t5_decoder_model(
    config: cfg.ModelConfig, embedding_layer, checkpoint_path: str
) -> nn.Module:
  model = T5Decoder(config, embedding_layer)
  decoder_tensor_names = ENCDEC_TENSOR_NAMES | {
      "cross_attn_query_proj": "{prefix}.block.{}.layer.1.EncDecAttention.q",
      "cross_attn_key_proj": "{prefix}.block.{}.layer.1.EncDecAttention.k",
      "cross_attn_value_proj": "{prefix}.block.{}.layer.1.EncDecAttention.v",
      "cross_attn_output_proj": "{prefix}.block.{}.layer.1.EncDecAttention.o",
      # In the decoder, the FF is layer 2 in the Transformer block
      "post_attn_norm": "{prefix}.block.{}.layer.2.layer_norm",
      # In the decoder, the cross attention is layer 1 in the Transformer block
      "pre_cross_attn_norm": "{prefix}.block.{}.layer.1.layer_norm",
  }

  decoder_tensor_names = {
      k: v.replace("{prefix}", "decoder").replace("{num}", "2")
      for k, v in decoder_tensor_names.items()
  }

  # Additional layer norms for Cross Attention in decoder
  # decoder_tensor_names["pre_attn_norm"] = "{prefix}.block.{}.layer.1.layer_norm",
  tensor_names = {
      "decoder.": loading_utils.ModelLoader.TensorNames(**decoder_tensor_names),
      "": loading_utils.ModelLoader.TensorNames(**TENSOR_NAMES),
  }
  loader = loading_utils.ModelLoader(checkpoint_path, names=tensor_names)
  # The embedding is shared between the encoder and decoder, so we set
  # strict=False.
  loader.load(model, strict=False, fuse_attention=False)
  return model


def get_sample_encoder_input_ids() -> torch.Tensor:
  idx = torch.tensor([[
      3856,
      27111,
      10,
      4425,
      51,
      4008,
      31,
      7,
      2306,
      16576,
      47,
      4381,
      16,
      8,
      3414,
      13,
      1410,
      16,
      932,
      11,
      1515,
      2766,
      6,
      11,
      4838,
      16,
      23964,
      16,
      1797,
      13,
      24,
      215,
      5,
      94,
      47,
      2017,
      168,
      1204,
      57,
      6800,
      7,
      11,
      9443,
      38,
      3673,
      8,
      4016,
      13,
      66,
      70,
      14234,
      5,
      2449,
      1215,
      83,
      17,
      16,
      8782,
      70,
      723,
      30,
      8,
      6162,
      13,
      1410,
      12,
      48,
      833,
      250,
      13,
      149,
      231,
      79,
      1858,
      16576,
      5,
      1,
  ]])
  return idx


def define_and_run_t5(checkpoint_path: str) -> None:
  current_dir = Path(__file__).parent.resolve()
  t5_goldens = torch.load(current_dir / "t5_lm_logits.pt")

  model = build_t5_model(checkpoint_path)

  idx = get_sample_encoder_input_ids()
  tokens = torch.full((1, 512), 0, dtype=torch.int, device="cpu")
  tokens[0, :77] = idx
  input_pos = torch.arange(0, 512, dtype=torch.int)

  decode_d_token = torch.tensor([[0]], dtype=torch.int)
  decode_d_input_pos = torch.tensor([0], dtype=torch.int)
  pad_mask = torch.zeros([model.config.kv_cache_max], dtype=torch.float32)
  pad_mask[77:] = float("-inf")
  lm_logits = model.forward(
      tokens, input_pos, decode_d_token, decode_d_input_pos, pad_mask
  )
  print("comparing with goldens..")
  assert torch.allclose(t5_goldens, lm_logits, atol=1e-05)


# TODO(haoliang): Move those tests.
def define_and_run_t5_split(checkpoint_path: str) -> None:
  current_dir = Path(__file__).parent.resolve()
  t5_goldens = torch.load(current_dir / "t5_lm_logits.pt")

  config = get_model_config_t5()
  embedding_layer = nn.Embedding(
      config.vocab_size, config.embedding_dim, padding_idx=0
  )
  t5_encoder_model = build_t5_encoder_model(
      config, embedding_layer, checkpoint_path
  )
  t5_decoder_model = build_t5_decoder_model(
      config, embedding_layer, checkpoint_path
  )
  idx = get_sample_encoder_input_ids()

  tokens = torch.full((1, 512), 0, dtype=torch.int, device="cpu")
  tokens[0, :77] = idx
  input_pos = torch.arange(0, 512, dtype=torch.int)

  decode_d_token = torch.tensor([[0]], dtype=torch.int)
  decode_d_input_pos = torch.tensor([0], dtype=torch.int)
  pad_mask = torch.zeros(
      [t5_encoder_model.config.kv_cache_max], dtype=torch.float32
  )
  pad_mask[77:] = float("-inf")
  hidden_states = t5_encoder_model.forward(tokens, input_pos, pad_mask)
  lm_logits = t5_decoder_model.forward(
      hidden_states, decode_d_token, decode_d_input_pos, pad_mask
  )
  print("comparing with goldens..")
  assert torch.allclose(t5_goldens, lm_logits, atol=1e-05)


if __name__ == "__main__":
  checkpoint = os.path.join(Path.home(), "Downloads/llm_data/t5")
  # define_and_run_t5(checkpoint)
  define_and_run_t5_split(checkpoint)
