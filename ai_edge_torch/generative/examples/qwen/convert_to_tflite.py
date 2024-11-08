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

"""Example of converting Qwen 2.5 models to multi-signature tflite model."""

import os
import pathlib

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.qwen import qwen
from ai_edge_torch.generative.utilities import converter

_MODEL_SIZE = flags.DEFINE_enum(
    'model_size',
    '3b',
    ['0.5b', '1.5b', '3b'],
    'The size of the model to convert.',
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    os.path.join(pathlib.Path.home(), 'Downloads/llm_data/qwen'),
    'The path to the model checkpoint, or directory holding the checkpoint.',
)
_TFLITE_PATH = flags.DEFINE_string(
    'tflite_path',
    '/tmp/',
    'The tflite file path to export.',
)
_PREFILL_SEQ_LEN = flags.DEFINE_integer(
    'prefill_seq_len',
    1024,
    'The maximum size of prefill input tensor.',
)
_KV_CACHE_MAX_LEN = flags.DEFINE_integer(
    'kv_cache_max_len',
    1280,
    'The maximum size of KV cache buffer, including both prefill and decode.',
)
_QUANTIZE = flags.DEFINE_bool(
    'quantize',
    True,
    'Whether the model should be quantized.',
)

_BUILDER = {
    '0.5b': qwen.build_0_5b_model,
    '1.5b': qwen.build_1_5b_model,
    '3b': qwen.build_3b_model,
}


def main(_):
  pytorch_model = _BUILDER[_MODEL_SIZE.value](
      _CHECKPOINT_PATH.value, kv_cache_max_len=_KV_CACHE_MAX_LEN.value
  )
  quant_suffix = 'q8' if _QUANTIZE.value else 'f32'
  model_size = _MODEL_SIZE.value.replace('.', '_')
  output_filename = f'qwen_{model_size}_{quant_suffix}_seq{_PREFILL_SEQ_LEN.value}_ekv{_KV_CACHE_MAX_LEN.value}.tflite'
  converter.convert_to_tflite(
      pytorch_model,
      tflite_path=os.path.join(_TFLITE_PATH.value, output_filename),
      prefill_seq_len=_PREFILL_SEQ_LEN.value,
      quantize=_QUANTIZE.value,
  )


if __name__ == '__main__':
  app.run(main)
