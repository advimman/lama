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

import dataclasses
import enum
from typing import Optional

from ai_edge_torch.generative.quantize import quant_recipe
from ai_edge_torch.quantize import pt2e_quantizer as pt2eq


@dataclasses.dataclass(frozen=True)
class QuantConfig:
  """Encapsulates a quantization configuration.

  Encapsulates all different quantization methods and schemes available for
  models converted with ai_edge_torch.

  Attributes:
    pt2e_quantizer: The instance of PT2EQuantizer used to quantize the model
      with PT2E quantization. This method of quantization is not applicable to
      models created with the Edge Generative API.
    generative_recipe: Quantization recipe to be applied on a model created with
      the Edge Generative API.
  """

  pt2e_quantizer: pt2eq.PT2EQuantizer = None
  generative_recipe: quant_recipe.GenerativeQuantRecipe = None

  @enum.unique
  class _QuantizerMode(enum.Enum):
    NONE = enum.auto()
    PT2E_DYNAMIC = enum.auto()
    PT2E_STATIC = enum.auto()
    AI_EDGE_QUANTIZER = enum.auto()

  _quantizer_mode: _QuantizerMode = _QuantizerMode.NONE

  def __init__(
      self,
      pt2e_quantizer: Optional[pt2eq.PT2EQuantizer] = None,
      generative_recipe: Optional[quant_recipe.GenerativeQuantRecipe] = None,
  ):
    """Initializes some internal states based on selected quantization method.

    Performs internal sanity checks to ensure that the user is inputting valid
    quantization requests. Verifies that the received quantization config
    is properly setup. Additionally sets up an utility enum _quantizer_mode to
    guide certain conversion processes.
    """
    if pt2e_quantizer is not None and generative_recipe is not None:
      raise ValueError('Cannot set both pt2e_quantizer and generative_recipe.')
    elif pt2e_quantizer is not None:
      object.__setattr__(self, 'pt2e_quantizer', pt2e_quantizer)
      object.__setattr__(
          self,
          '_quantizer_mode',
          (
              self._QuantizerMode.PT2E_DYNAMIC
              if pt2e_quantizer.global_config.is_dynamic
              else self._QuantizerMode.PT2E_STATIC
          ),
      )
    elif generative_recipe is not None:
      generative_recipe.verify()
      object.__setattr__(self, 'generative_recipe', generative_recipe)
      object.__setattr__(
          self, '_quantizer_mode', self._QuantizerMode.AI_EDGE_QUANTIZER
      )
    else:
      raise ValueError(
          'Either pt2e_quantizer or generative_recipe must be set.'
      )
