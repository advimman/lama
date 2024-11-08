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

from ai_edge_quantizer import quantizer
from ai_edge_torch.generative.quantize import quant_attrs
from ai_edge_torch.generative.quantize import quant_recipe

_ComputePrecision = quantizer.qtyping.ComputePrecision
_QuantGranularity = quantizer.qtyping.QuantGranularity
_OpName = quantizer.qtyping.TFLOperationName
_TensorQuantConfig = quantizer.qtyping.TensorQuantizationConfig
_OpQuantConfig = quantizer.qtyping.OpQuantizationConfig

_DEFAULT_REGEX_STR = '.*'
_SINGULAR_TRANSFORMER_BLOCK_REGEX_STR = 'transformer_block'
_IDX_TRANSFORMER_BLOCKS_REGEX_STR = 'transformer_blocks\[{}\]'
_ATTENTION_REGEX_STR = 'ai_edge_torch.generative.layers.attention'
_FEEDFORWARD_REGEX_STR = 'ai_edge_torch.generative.layers.feed_forward'
_EMBEDDING_REGEX_STR = 'Embedding_tok_embedding'
_ANY_TWO_DIGITS_REGEX_STR = '\d{1,2}'


def _get_nbits_from_dtype(dtype: quant_attrs.Dtype) -> int:
  if dtype == quant_attrs.Dtype.FP32:
    return 32
  elif dtype == quant_attrs.Dtype.FP16:
    return 16
  elif dtype == quant_attrs.Dtype.INT8:
    return 8
  raise ValueError('Unimplemented number of bits')


def _get_dtype_from_dtype(
    dtype: quant_attrs.Dtype,
) -> quantizer.qtyping.TensorDataType:
  if dtype == quant_attrs.Dtype.FP32 or dtype == quant_attrs.Dtype.FP16:
    return quantizer.qtyping.TensorDataType.FLOAT
  else:
    return quantizer.qtyping.TensorDataType.INT


def _get_compute_precision_from_mode(
    mode: quant_attrs.Mode,
) -> _ComputePrecision:
  if mode == quant_attrs.Mode.DYNAMIC_RANGE:
    return _ComputePrecision.INTEGER
  elif mode == quant_attrs.Mode.WEIGHT_ONLY:
    return _ComputePrecision.FLOAT
  raise ValueError('Unimplemented execution mode')


def _get_explicit_dequant_from_mode(mode: quant_attrs.Mode) -> bool:
  if mode == quant_attrs.Mode.DYNAMIC_RANGE:
    return False
  elif mode == quant_attrs.Mode.WEIGHT_ONLY:
    return True
  raise ValueError('Unimplemented execution mode')


def _get_granularity(
    granularity: quant_attrs.Granularity,
) -> bool:
  if granularity == quant_attrs.Granularity.CHANNELWISE:
    return _QuantGranularity.CHANNELWISE
  if granularity == quant_attrs.Granularity.NONE:
    return _QuantGranularity.TENSORWISE
  raise ValueError('Unimplemented granularity')


def _get_algorithm_key_from_algorithm(algo: quant_attrs.Algorithm) -> str:
  if algo == quant_attrs.Algorithm.MIN_MAX:
    return quantizer.algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT
  elif algo == quant_attrs.Algorithm.FLOAT_CAST:
    return quantizer.algorithm_manager.AlgorithmName.FLOAT_CASTING
  raise ValueError('Unimplemented algorithm')


def _set_quant_config(
    rm: quantizer.recipe_manager.RecipeManager,
    layer_recipe: quant_recipe.LayerQuantRecipe,
    regex: str,
):
  rm.add_quantization_config(
      regex=regex,
      operation_name=_OpName.ALL_SUPPORTED,
      op_config=_OpQuantConfig(
          weight_tensor_config=_TensorQuantConfig(
              num_bits=_get_nbits_from_dtype(layer_recipe.weight_dtype),
              symmetric=True,
              granularity=_get_granularity(layer_recipe.granularity),
              dtype=_get_dtype_from_dtype(layer_recipe.weight_dtype),
          ),
          compute_precision=_get_compute_precision_from_mode(layer_recipe.mode),
          explicit_dequantize=_get_explicit_dequant_from_mode(
              layer_recipe.mode
          ),
      ),
      algorithm_key=_get_algorithm_key_from_algorithm(layer_recipe.algorithm),
  )


def translate_to_ai_edge_recipe(
    recipe: quant_recipe.GenerativeQuantRecipe,
) -> quantizer.recipe_manager.ModelQuantizationRecipe:
  rm = quantizer.recipe_manager.RecipeManager()

  if recipe.default is not None:
    _set_quant_config(rm, recipe.default, _DEFAULT_REGEX_STR)

  if recipe.embedding is not None:
    _set_quant_config(rm, recipe.embedding, _EMBEDDING_REGEX_STR)

  if recipe.attention is not None:
    if isinstance(recipe.attention, dict):
      for idx, layer in recipe.attention.items():
        _set_quant_config(
            rm,
            layer,
            f'{_IDX_TRANSFORMER_BLOCKS_REGEX_STR.format(idx)}/{_ATTENTION_REGEX_STR}',
        )
    else:
      _set_quant_config(
          rm,
          recipe.attention,
          f'{_SINGULAR_TRANSFORMER_BLOCK_REGEX_STR}/{_ATTENTION_REGEX_STR}',
      )

  if recipe.feedforward is not None:
    if isinstance(recipe.feedforward, dict):
      for idx, layer in recipe.feedforward.items():
        _set_quant_config(
            rm,
            layer,
            f'{_IDX_TRANSFORMER_BLOCKS_REGEX_STR.format(idx)}/{_FEEDFORWARD_REGEX_STR}',
        )
    else:
      _set_quant_config(
          rm,
          recipe.feedforward,
          f'{_SINGULAR_TRANSFORMER_BLOCK_REGEX_STR}/{_FEEDFORWARD_REGEX_STR}',
      )

  return rm.get_quantization_recipe()


def quantize_model(
    model: bytes, recipe: quantizer.recipe_manager.ModelQuantizationRecipe
) -> bytearray:
  qt = quantizer.Quantizer(model, recipe)
  result = qt.quantize()
  return result.quantized_model
