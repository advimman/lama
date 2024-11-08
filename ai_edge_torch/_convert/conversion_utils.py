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

from typing import Any

from ai_edge_torch.quantize import quant_config as qcfg
import tensorflow as tf


def apply_tfl_converter_flags(
    converter: tf.lite.TFLiteConverter, tfl_converter_flags: dict[str, Any]
):
  """Applies TFLite converter flags to the converter.

  Args:
    converter: TFLite converter.
    tfl_converter_flags: TFLite converter flags.
  """

  def _set_converter_flag(path: list[Any]):
    if len(path) < 2:
      raise ValueError("Expecting at least two values in the path.")

    target_obj = converter
    for idx in range(len(path) - 2):
      target_obj = getattr(target_obj, path[idx])

    setattr(target_obj, path[-2], path[-1])

  def _iterate_dict_tree(flags_dict: dict[str, Any], path: list[Any]):
    for key, value in flags_dict.items():
      path.append(key)
      if isinstance(value, dict):
        _iterate_dict_tree(value, path)
      else:
        path.append(value)
        _set_converter_flag(path)
        path.pop()
      path.pop()

  _iterate_dict_tree(tfl_converter_flags, [])


def set_tfl_converter_quant_flags(
    converter: tf.lite.TFLiteConverter, quant_config: qcfg.QuantConfig
):
  if quant_config is not None:
    quantizer_mode = quant_config._quantizer_mode
    if quantizer_mode == qcfg.QuantConfig._QuantizerMode.PT2E_DYNAMIC:
      converter._experimental_qdq_conversion_mode = "DYNAMIC"
    elif quantizer_mode == qcfg.QuantConfig._QuantizerMode.PT2E_STATIC:
      converter._experimental_qdq_conversion_mode = "STATIC"
