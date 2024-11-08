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

"""Common utility functions for model conversion."""

import ai_edge_torch
from ai_edge_torch._convert import converter as converter_utils
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.quantize import quant_recipes
import torch


def convert_to_tflite(
    pytorch_model: torch.nn.Module,
    tflite_path: str,
    prefill_seq_len: int = 512,
    quantize: bool = True,
):
  """Converts a nn.Module model to multi-signature tflite model.

  A PyTorch model will be converted to a tflite model with two signatures:
  "prefill" and "decode".

  "prefill" signature takes a tensor of shape [1, prefill_seq_len] of token
  sequence, a tensor of shape [1, prefill_seq_len] of token positions, and an
  external KV cache as a sample input.

  "decode" signature takes a tensor of shape [1, 1] of token sequence, a tensor
  of shape [1, 1] of the token position, and an external KV cache as a sample
  input.

  The final tflite model will be exported to tflite_path.

  Args:
      pytorch_model (torch.nn.Module): PyTorch model to convert to tflite.
      tflite_path (str): The tflite file path to export.
      prefill_seq_len (int, optional): The maximum size of prefill input tensor.
        Defaults to 512.
      quantize (bool, optional): Whether the model should be quanized. Defaults
        to True.
  """
  # Tensors used to trace the model graph during conversion.
  prefill_tokens = torch.full((1, prefill_seq_len), 0, dtype=torch.int)
  prefill_input_pos = torch.arange(0, prefill_seq_len, dtype=torch.int)
  decode_token = torch.tensor([[0]], dtype=torch.int)
  decode_input_pos = torch.tensor([0], dtype=torch.int)
  kv = kv_utils.KVCache.from_model_config(pytorch_model.config)

  quant_config = quant_recipes.full_int8_dynamic_recipe() if quantize else None
  edge_model = (
      ai_edge_torch.signature(
          'prefill',
          pytorch_model,
          sample_kwargs={
              'tokens': prefill_tokens,
              'input_pos': prefill_input_pos,
              'kv_cache': kv,
          },
      )
      .signature(
          'decode',
          pytorch_model,
          sample_kwargs={
              'tokens': decode_token,
              'input_pos': decode_input_pos,
              'kv_cache': kv,
          },
      )
      .convert(quant_config=quant_config)
  )
  edge_model.export(tflite_path)


def convert_to_tflite_multi_prefill_lens(
    pytorch_model: torch.nn.Module,
    tflite_path: str,
    prefill_seq_lens: list[int],
    quantize: bool = True,
):
  """Converts a nn.Module model to multi-signature tflite model with different

  prefill lengths.

  A PyTorch model will be converted to a tflite model with several signatures:
  "prefill_[prefill_seq_len]" and "decode".

  "prefill_[prefill_seq_len]" signature takes a tensor of shape [1,
  prefill_seq_len] of token
  sequence, a tensor of shape [1, prefill_seq_len] of token positions, and an
  external KV cache as a sample input.

  "decode" signature takes a tensor of shape [1, 1] of token sequence, a tensor
  of shape [1, 1] of the token position, and an external KV cache as a sample
  input.

  The final tflite model will be exported to tflite_path.

  Args:
      pytorch_model (torch.nn.Module): PyTorch model to convert to tflite.
      tflite_path (str): The tflite file path to export.
      prefill_seq_lens (list[int]): A list of prefill lengths to export.
      quantize (bool, optional): Whether the model should be quanized. Defaults
        to True.
  """
  # Tensors used to trace the model graph during conversion.
  prefill_tokens_list = []
  prefill_input_pos_list = []
  for prefill_seq_len in prefill_seq_lens:
    prefill_tokens_list.append(
        torch.full((1, prefill_seq_len), 0, dtype=torch.int)
    )
    prefill_input_pos_list.append(
        torch.arange(0, prefill_seq_len, dtype=torch.int)
    )

  decode_token = torch.tensor([[0]], dtype=torch.int)
  decode_input_pos = torch.tensor([0], dtype=torch.int)
  kv = kv_utils.KVCache.from_model_config(pytorch_model.config)

  quant_config = quant_recipes.full_int8_dynamic_recipe() if quantize else None
  converter = converter_utils.Converter()
  for i in range(len(prefill_seq_lens)):
    prefill_seq_len = prefill_seq_lens[i]
    prefill_tokens = prefill_tokens_list[i]
    prefill_input_pos = prefill_input_pos_list[i]
    converter.add_signature(
        f'prefill_{prefill_seq_len}',
        pytorch_model,
        sample_kwargs={
            'tokens': prefill_tokens,
            'input_pos': prefill_input_pos,
            'kv_cache': kv,
        },
    )

  edge_model = converter.add_signature(
      'decode',
      pytorch_model,
      sample_kwargs={
          'tokens': decode_token,
          'input_pos': decode_input_pos,
          'kv_cache': kv,
      },
  ).convert(quant_config=quant_config)
  edge_model.export(tflite_path)
