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

from dataclasses import dataclass
from typing import Optional, Union

from ai_edge_torch.generative.quantize import quant_attrs
from ai_edge_torch.generative.quantize import supported_schemes


@dataclass
class LayerQuantRecipe:
  """Quantization recipe for a single Edge Generative API layer (e.g. Attention).

  Generic layer-scoped quantization recipe that specifies how this layer should
  be quantized by the Edge Generative API. This is applicable to layers
  implemented
  in ai_edge_torch/generative/layers/. Combinations of attributes that are not
  supported during runtime will be detected when .verify() is called.

  Attributes:
    activation_dtype: Desired data type of activation tensors.
    weight_dtype: Desired data type of weight tensors.
    mode: Type of quantization.
    algorithm: Algorithm for calculating quantization parameters.
    granularity: Granularity of quantization.
  """

  activation_dtype: quant_attrs.Dtype
  weight_dtype: quant_attrs.Dtype
  mode: quant_attrs.Mode
  algorithm: quant_attrs.Algorithm
  granularity: quant_attrs.Granularity

  def __str__(self):
    return (
        f'(a:{self.activation_dtype.name}, '
        f'w:{self.weight_dtype.name}, '
        f'{self.mode.name}, '
        f'{self.algorithm.name}, '
        f'{self.granularity.name})'
    )

  __repr__ = __str__

  def verify(self):
    """Checks if all attributes configured are supported in runtime.

    Raises:
      ValueError: If any attributes are incompatible.
    """
    is_valid = False
    for supported in supported_schemes.get_supported_layer_schemes():
      if (
          self.activation_dtype == supported[0]
          and self.weight_dtype == supported[1]
          and self.mode == supported[2]
          and self.algorithm == supported[3]
          and self.granularity == supported[4]
      ):
        is_valid = True
        break

    if not is_valid:
      raise ValueError(
          'Unsupported LayerQuantRecipe configuration. See'
          ' get_supported_recipe_matrix()'
      )


@dataclass
class GenerativeQuantRecipe:
  """Quantization recipe for a model composed of the Edge Generative API layers.

  Some layers can be specified with different `LayerQuantRecipe` for each block
  by
  providing a dictionary keyed by the TransformerBlock index, e.g. attention
  and feedforward. For example,

  ```
  default = LayerQuantRecipeA
  attention = { 2: LayerQuantRecipeB }
  feedforward = { 3: LayerQuantRecipeC }
  ```

  will apply LayerQuantRecipeA to the entire model, overriden by
  LayerQuantRecipeB for the TransformerBlock[2].attention layer and
  LayerQuantRecipeC for the TransformerBlock[3].feedforward layer. Any config
  with invalid indices will be ignored.

  Attributes:
    default: The quantization recipe for global scope of the model.
    embedding: Recipe for the embedding table.
    attention: Recipe for the attention blocks. This could be specified with
      different LayerQuantRecipe for each block by providing a dictionary keyed
      by the TransformerBlock index.
    feedforward: Recipe for the feedforward layers. This could be specified with
      different LayerQuantRecipe for each block by providing a dictionary keyed
      by the TransformerBlock index.
  """

  default: Optional[LayerQuantRecipe] = None
  embedding: Optional[LayerQuantRecipe] = None
  attention: Union[
      Optional[LayerQuantRecipe], Optional[dict[int, LayerQuantRecipe]]
  ] = None
  feedforward: Union[
      Optional[LayerQuantRecipe], Optional[dict[int, LayerQuantRecipe]]
  ] = None

  def __str__(self):
    return f"""GenerativeQuantRecipe(
  Default: {self.default}
  Embedding: {self.embedding}
  Attention: {self.attention}
  Feedforward: {self.feedforward}
)"""

  __repr__ = __str__

  def verify(self):
    """Checks if the recipe configured can be supported in runtime.

    Raises:
      ValueError: If the recipe configured is invalid or unsupported.
    """
    if self.default is not None:
      self.default.verify()
    if self.embedding is not None:
      self.embedding.verify()
    if self.attention is not None:
      if isinstance(self.attention, dict):
        for recipe in self.attention.values():
          recipe.verify()
      else:
        self.attention.verify()
    if self.feedforward is not None:
      if isinstance(self.feedforward, dict):
        for recipe in self.feedforward.values():
          recipe.verify()
      else:
        self.feedforward.verify()
