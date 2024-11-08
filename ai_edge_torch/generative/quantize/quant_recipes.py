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

"""Helper functions to create common and supported quantization recipes.

These recipes will work with models created with the Edge Generative API only.
Assume Transformer architecture congruent with
ai_edge_torch/generative/layers/model_config.py:ModelConfig.

Typical usage example:

  quant_config = quant_recipes.full_int8_dynamic_recipe()
  edge_model = ai_edge_torch.convert(
      model, (tokens, input_pos), quant_config=quant_config
  )
"""

from ai_edge_torch.generative.quantize import quant_recipe
from ai_edge_torch.generative.quantize import quant_recipe_utils
from ai_edge_torch.quantize import quant_config


def full_int8_dynamic_recipe() -> quant_config.QuantConfig:
  return quant_config.QuantConfig(
      generative_recipe=quant_recipe.GenerativeQuantRecipe(
          default=quant_recipe_utils.create_layer_quant_int8_dynamic(),
      )
  )


def full_int8_weight_only_recipe() -> quant_config.QuantConfig:
  return quant_config.QuantConfig(
      generative_recipe=quant_recipe.GenerativeQuantRecipe(
          default=quant_recipe_utils.create_layer_quant_int8_weight_only(),
      )
  )


def full_fp16_recipe() -> quant_config.QuantConfig:
  return quant_config.QuantConfig(
      generative_recipe=quant_recipe.GenerativeQuantRecipe(
          default=quant_recipe_utils.create_layer_quant_fp16()
      )
  )
