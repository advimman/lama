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

import ai_edge_torch
from ai_edge_torch import config
from ai_edge_torch.generative.examples.test_models import toy_model  # NOQA
from ai_edge_torch.generative.quantize import quant_recipe
from ai_edge_torch.generative.quantize import quant_recipe_utils
from ai_edge_torch.generative.quantize import quant_recipes
from ai_edge_torch.generative.quantize.quant_attrs import Algorithm
from ai_edge_torch.generative.quantize.quant_attrs import Dtype
from ai_edge_torch.generative.quantize.quant_attrs import Granularity
from ai_edge_torch.generative.quantize.quant_attrs import Mode
from ai_edge_torch.quantize import quant_config
from ai_edge_torch.testing import model_coverage
import torch

from absl.testing import absltest as googletest
from absl.testing import parameterized


class TestVerifyRecipes(parameterized.TestCase):
  """Unit tests that check for model quantization recipes."""

  @parameterized.parameters([
      (Dtype.FP32, Dtype.FP32),
      (Dtype.INT8, Dtype.INT8),
      (Dtype.INT8, Dtype.FP16),
      (Dtype.FP16, Dtype.INT8),
      (Dtype.FP16, Dtype.FP16),
  ])
  def test_verify_invalid_recipes(
      self,
      activation,
      weight,
  ):
    for m in Mode:
      for a in Algorithm:
        for g in Granularity:
          with self.assertRaises(ValueError):
            quant_recipe.LayerQuantRecipe(activation, weight, m, a, g).verify()

  @parameterized.parameters([
      (
          Dtype.FP32,
          Dtype.INT8,
          Mode.DYNAMIC_RANGE,
          Algorithm.MIN_MAX,
          Granularity.CHANNELWISE,
      ),
      (
          Dtype.FP32,
          Dtype.INT8,
          Mode.WEIGHT_ONLY,
          Algorithm.MIN_MAX,
          Granularity.CHANNELWISE,
      ),
      (
          Dtype.FP32,
          Dtype.FP16,
          Mode.WEIGHT_ONLY,
          Algorithm.FLOAT_CAST,
          Granularity.NONE,
      ),
  ])
  def test_verify_valid_recipes(
      self,
      activation,
      weight,
      mode,
      algo,
      granularity,
  ):
    quant_recipe.LayerQuantRecipe(
        activation, weight, mode, algo, granularity
    ).verify()


class TestQuantizeConvert(parameterized.TestCase):
  """Test conversion with quantization."""

  def _attention_int8_dynamic_recipe() -> quant_config.QuantConfig:
    return quant_config.QuantConfig(
        generative_recipe=quant_recipe.GenerativeQuantRecipe(
            attention=quant_recipe_utils.create_layer_quant_int8_dynamic(),
        )
    )

  def _feedforward_int8_dynamic_recipe() -> quant_config.QuantConfig:
    return quant_config.QuantConfig(
        generative_recipe=quant_recipe.GenerativeQuantRecipe(
            feedforward=quant_recipe_utils.create_layer_quant_int8_dynamic(),
        )
    )

  @parameterized.parameters([
      (quant_recipes.full_fp16_recipe()),
      (quant_recipes.full_int8_dynamic_recipe()),
      (quant_recipes.full_int8_weight_only_recipe()),
      (_attention_int8_dynamic_recipe()),
      (_feedforward_int8_dynamic_recipe()),
  ])
  def test_quantize_convert_toy_sizes(self, quant_config):
    config = toy_model.get_model_config()
    pytorch_model = toy_model.ToySingleLayerModel(config)
    idx = torch.unsqueeze(torch.arange(0, 100, dtype=torch.int), 0)
    input_pos = torch.arange(0, 100, dtype=torch.int)

    quantized_model = ai_edge_torch.convert(
        pytorch_model, (idx, input_pos), quant_config=quant_config
    )
    float_model = ai_edge_torch.convert(pytorch_model, (idx, input_pos))
    self.assertLess(
        len(quantized_model._tflite_model),
        len(float_model._tflite_model),
        "Quantized model isn't smaller than F32 model.",
    )

  def test_quantize_convert_toy_weight_sharing(self):
    config = toy_model.get_model_config()
    pytorch_model = toy_model.ToySingleLayerModelWeightSharing(config)
    idx = torch.unsqueeze(torch.arange(0, 100, dtype=torch.int), 0)
    input_pos = torch.arange(0, 100, dtype=torch.int)

    quant_config = quant_recipes.full_int8_dynamic_recipe()
    quantized_model = ai_edge_torch.convert(
        pytorch_model, (idx, input_pos), quant_config=quant_config
    )
    float_model = ai_edge_torch.convert(pytorch_model, (idx, input_pos))
    self.assertLess(
        len(quantized_model._tflite_model),
        len(float_model._tflite_model),
        "Quantized model isn't smaller than F32 model.",
    )

  def test_quantize_convert_compare_toy(self):
    self.skipTest("b/338288901")
    config = toy_model_with_kv_cache.get_model_config()
    pytorch_model = toy_model_with_kv_cache.ToyModelWithKV(config)
    idx, input_pos = torch.tensor([[1]], dtype=torch.int), torch.tensor(
        [10], dtype=torch.int64
    )

    quant_config = quant_recipes.full_fp16_recipe()
    quantized_model = ai_edge_torch.convert(
        pytorch_model, (idx, input_pos), quant_config=quant_config
    )
    float_model = ai_edge_torch.convert(pytorch_model, (idx, input_pos))

    self.assertLess(
        len(quantized_model._tflite_model), len(float_model._tflite_model)
    )
    self.assertTrue(
        model_coverage.compare_tflite_torch(
            quantized_model,
            pytorch_model,
            (idx, input_pos),
            num_valid_inputs=1,
            atol=1e-3,
            rtol=1e-3,
        )
    )


if __name__ == "__main__":
  googletest.main()
