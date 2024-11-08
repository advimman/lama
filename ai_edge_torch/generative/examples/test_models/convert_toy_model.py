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
# A toy example which has a single-layer transformer block.
from absl import app
import ai_edge_torch
from ai_edge_torch import lowertools
from ai_edge_torch.generative.examples.test_models import toy_model
from ai_edge_torch.generative.examples.test_models import toy_model_with_kv_cache
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import torch

KV_CACHE_MAX_LEN = 100


def convert_toy_model(_) -> None:
  """Converts a toy model to tflite."""
  model = toy_model.ToySingleLayerModel(toy_model.get_model_config())
  idx = torch.unsqueeze(torch.arange(0, KV_CACHE_MAX_LEN), 0)
  input_pos = torch.arange(0, KV_CACHE_MAX_LEN)
  print('running an inference')
  print(
      model.forward(
          idx,
          input_pos,
      )
  )

  # Convert model to tflite.
  print('converting model to tflite')
  edge_model = ai_edge_torch.convert(
      model,
      (
          idx,
          input_pos,
      ),
  )
  edge_model.export('/tmp/toy_model.tflite')


def _export_stablehlo_mlir(model, args):
  ep = torch.export.export(model, args)
  return lowertools.exported_program_to_mlir_text(ep)


def convert_toy_model_with_kv_cache(_) -> None:
  """Converts a toy model with kv cache to tflite."""
  dump_mlir = False

  config = toy_model_with_kv_cache.get_model_config()
  model = toy_model_with_kv_cache.ToyModelWithKVCache(config)
  model.eval()
  print('running an inference')
  kv = kv_utils.KVCache.from_model_config(config)

  tokens, input_pos = toy_model_with_kv_cache.get_sample_prefill_inputs()
  decode_token, decode_input_pos = (
      toy_model_with_kv_cache.get_sample_decode_inputs()
  )
  print(model.forward(tokens, input_pos, kv))

  if dump_mlir:
    mlir_text = _export_stablehlo_mlir(model, (tokens, input_pos, kv))
    with open('/tmp/toy_model_with_external_kv.stablehlo.mlir', 'w') as f:
      f.write(mlir_text)

  # Convert model to tflite with 2 signatures (prefill + decode).
  print('converting toy model to tflite with 2 signatures (prefill + decode)')
  edge_model = (
      ai_edge_torch.signature(
          'prefill',
          model,
          sample_kwargs={
              'tokens': tokens,
              'input_pos': input_pos,
              'kv_cache': kv,
          },
      )
      .signature(
          'decode',
          model,
          sample_kwargs={
              'tokens': decode_token,
              'input_pos': decode_input_pos,
              'kv_cache': kv,
          },
      )
      .convert()
  )
  edge_model.export('/tmp/toy_external_kv_cache.tflite')


if __name__ == '__main__':
  app.run(convert_toy_model)
