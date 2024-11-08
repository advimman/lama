# Copyright 2024 The AI Edge Torch Authors. All Rights Reserved.
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

"""Helper functions to construct custom quantization recipes.

These are intended for more advanced users who want to configure their own
quantization recipes. For pre-constructed recipes, use `quant_recipes.py`
instead.

Typical usage example:

1. Applying a single layer recipe to the entire model

  quant_recipe.GenerativeQuantRecipe(
    default=quant_recipe_utils.create_layer_quant_int8_dynamic()
  )
"""

from ai_edge_torch.generative.quantize import quant_attrs
from ai_edge_torch.generative.quantize import quant_recipe


def create_layer_quant_int8_dynamic() -> quant_recipe.LayerQuantRecipe:
  return quant_recipe.LayerQuantRecipe(
      activation_dtype=quant_attrs.Dtype.FP32,
      weight_dtype=quant_attrs.Dtype.INT8,
      mode=quant_attrs.Mode.DYNAMIC_RANGE,
      algorithm=quant_attrs.Algorithm.MIN_MAX,
      granularity=quant_attrs.Granularity.CHANNELWISE,
  )


def create_layer_quant_int8_weight_only() -> quant_recipe.LayerQuantRecipe:
  return quant_recipe.LayerQuantRecipe(
      activation_dtype=quant_attrs.Dtype.FP32,
      weight_dtype=quant_attrs.Dtype.INT8,
      mode=quant_attrs.Mode.WEIGHT_ONLY,
      algorithm=quant_attrs.Algorithm.MIN_MAX,
      granularity=quant_attrs.Granularity.CHANNELWISE,
  )


def create_layer_quant_fp16() -> quant_recipe.LayerQuantRecipe:
  return quant_recipe.LayerQuantRecipe(
      activation_dtype=quant_attrs.Dtype.FP32,
      weight_dtype=quant_attrs.Dtype.FP16,
      mode=quant_attrs.Mode.WEIGHT_ONLY,
      algorithm=quant_attrs.Algorithm.FLOAT_CAST,
      granularity=quant_attrs.Granularity.NONE,
  )
