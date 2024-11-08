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

import os
from pathlib import Path

import ai_edge_torch
from ai_edge_torch.generative.examples.t5 import t5
from ai_edge_torch.generative.quantize import quant_recipes
import numpy as np
import torch


# TODO(haoliang): clean this up untile 2-sig model is validated e2e.
def convert_t5_to_tflite_singlesig(checkpoint_path: str):
  pytorch_model = t5.build_t5_model(checkpoint_path)

  # encoder
  seq_len = 512
  prefill_e_tokens = torch.full((1, seq_len), 0, dtype=torch.int)
  prompt_e_token = [1, 2, 3, 4, 5, 6]
  prefill_e_tokens[0, : len(prompt_e_token)] = torch.tensor(
      prompt_e_token, dtype=torch.int
  )
  prefill_e_input_pos = torch.arange(0, seq_len, dtype=torch.int)
  prefill_d_tokens = torch.full((1, seq_len), 0, dtype=torch.int)
  prompt_d_token = [1, 2, 3, 4, 5, 6]
  prefill_d_tokens[0, : len(prompt_d_token)] = torch.tensor(
      prompt_d_token, dtype=torch.int
  )
  prefill_d_input_pos = torch.arange(0, seq_len, dtype=torch.int)

  # decoder
  decode_token = torch.tensor([[1]], dtype=torch.int)
  decode_input_pos = torch.tensor([0], dtype=torch.int)
  decode_d_token = torch.tensor([[1]], dtype=torch.int)
  decode_d_input_pos = torch.tensor([0], dtype=torch.int)

  # Pad mask for self attention only on "real" tokens.
  # Pad with `-inf` for any tokens indices that aren't desired.
  pad_mask = torch.zeros([seq_len], dtype=torch.float32)

  edge_model = ai_edge_torch.signature(
      'decode',
      pytorch_model,
      (
          prefill_e_tokens,
          prefill_e_input_pos,
          decode_d_token,
          decode_d_input_pos,
          pad_mask,
      ),
  ).convert()

  edge_model.export('/tmp/t5_encode_decode.tflite')


def convert_t5_to_tflite_multisig(checkpoint_path: str):
  config = t5.get_model_config_t5()
  embedding_layer = torch.nn.Embedding(
      config.vocab_size, config.embedding_dim, padding_idx=0
  )
  t5_encoder_model = t5.build_t5_encoder_model(
      config, embedding_layer, checkpoint_path
  )
  t5_decoder_model = t5.build_t5_decoder_model(
      config, embedding_layer, checkpoint_path
  )

  # encoder
  seq_len = 512
  prefill_e_tokens = torch.full((1, seq_len), 0, dtype=torch.int)
  prompt_e_token = [1, 2, 3, 4, 5, 6]
  prefill_e_tokens[0, : len(prompt_e_token)] = torch.tensor(
      prompt_e_token, dtype=torch.int
  )
  prefill_e_input_pos = torch.arange(0, seq_len, dtype=torch.int)
  prefill_d_tokens = torch.full((1, seq_len), 0, dtype=torch.int)
  prompt_d_token = [1, 2, 3, 4, 5, 6]
  prefill_d_tokens[0, : len(prompt_d_token)] = torch.tensor(
      prompt_d_token, dtype=torch.int
  )
  prefill_d_input_pos = torch.arange(0, seq_len, dtype=torch.int)

  # decoder
  decode_token = torch.tensor([[1]], dtype=torch.int)
  decode_input_pos = torch.tensor([0], dtype=torch.int)
  decode_d_token = torch.tensor([[1]], dtype=torch.int)
  decode_d_input_pos = torch.tensor([0], dtype=torch.int)

  # Pad mask for self attention only on "real" tokens.
  # Pad with `-inf` for any tokens indices that aren't desired.
  pad_mask = torch.zeros([seq_len], dtype=torch.float32)
  hidden_states = torch.zeros((1, 512, 768), dtype=torch.float32)
  quant_config = quant_recipes.full_int8_dynamic_recipe()

  edge_model = (
      ai_edge_torch.signature(
          'encode',
          t5_encoder_model,
          (
              prefill_e_tokens,
              prefill_e_input_pos,
              pad_mask,
          ),
      )
      .signature(
          'decode',
          t5_decoder_model,
          (
              hidden_states,
              decode_d_token,
              decode_d_input_pos,
              pad_mask,
          ),
      )
      .convert(quant_config=quant_config)
  )

  edge_model.export('/tmp/t5_encode_decode_2_sigs.tflite')


if __name__ == '__main__':
  checkpoint_path = os.path.join(Path.home(), 'Downloads/llm_data/t5')
  # convert_t5_to_tflite_singlesig(checkpoint_path)
  convert_t5_to_tflite_multisig(checkpoint_path)
