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

import enum


@enum.unique
class Dtype(enum.Enum):
  """Data types and precision of tensors."""

  FP32 = enum.auto()
  FP16 = enum.auto()
  INT8 = enum.auto()


@enum.unique
class Algorithm(enum.Enum):
  """Algorithm used to calculate quantization parameters.

  Attributes:
    MIN_MAX: Maps the min/max of floating point space to the min/max of
      quantized space and quantize uniformly.
    FLOAT_CAST: Casts a float to another float of a different type.
  """

  MIN_MAX = enum.auto()
  FLOAT_CAST = enum.auto()


@enum.unique
class Mode(enum.Enum):
  """Mode of quantization.

  Attributes:
    DYNAMIC_RANGE: Quantize activations during runtime and weights statically to
      perform computation in integers.
    WEIGHT_ONLY: Quantize weights statically and dequantize during runtime to
      perform computation in floating points.
  """

  DYNAMIC_RANGE = enum.auto()
  WEIGHT_ONLY = enum.auto()


@enum.unique
class Granularity(enum.Enum):
  """Granularity of quantization parameters.

  Attributes:
    NONE: Granularity not applicable to this quantization scheme.
    CHANNELWISE: Or per-channel quantization. Each channel of relevant tensors
      is quantized independently of one another.
  """

  NONE = enum.auto()
  CHANNELWISE = enum.auto()
