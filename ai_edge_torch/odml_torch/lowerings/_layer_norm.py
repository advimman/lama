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
"""Provides lowering for coreaten to stablehlo for LayerNorm."""

import math
from typing import Optional
from ai_edge_torch.odml_torch.lowerings import registry
from ai_edge_torch.odml_torch.lowerings import utils
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch


# native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight,
# Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
@registry.lower(torch.ops.aten.native_layer_norm)
def _aten_native_layer_norm(
    lctx,
    data: ir.Value,
    normalized_shape: list[int],
    weight: Optional[ir.Value],
    bias: Optional[ir.Value],
    eps: float,
):
  data_type: ir.RankedTensorType = data.type
  unnormalized_count = math.prod(data_type.shape) // math.prod(normalized_shape)
  dest_shape = [
      1,
      unnormalized_count,
      math.prod(normalized_shape),
  ]
  dest_type = ir.RankedTensorType.get(dest_shape, data_type.element_type)

  reshaped_data = stablehlo.reshape(dest_type, data)

  one = utils.splat(1, data_type.element_type, [unnormalized_count])
  zero = utils.splat(0, data_type.element_type, [unnormalized_count])
  output, mean, var = stablehlo.batch_norm_training(
      reshaped_data, one, zero, eps, 1
  )
  eps_splat = utils.splat(eps, var.type.element_type, var.type.shape)
  rstd = stablehlo.rsqrt(stablehlo.add(var, eps_splat))

  stats_shape = data_type.shape[: -1 * len(normalized_shape)] + [1] * len(
      normalized_shape
  )
  stats_type = ir.RankedTensorType.get(stats_shape, data_type.element_type)
  mean = stablehlo.reshape(stats_type, mean)
  rstd = stablehlo.reshape(stats_type, rstd)

  output = stablehlo.reshape(data_type, output)

  data_rank = len(data_type.shape)
  normalized_rank = len(normalized_shape)
  if weight is not None:
    weight = stablehlo.broadcast_in_dim(
        data_type, weight, list(range(data_rank - normalized_rank, data_rank))
    )
    output = stablehlo.multiply(weight, output)
  if bias is not None:
    bias = stablehlo.broadcast_in_dim(
        data_type, bias, list(range(data_rank - normalized_rank, data_rank))
    )
    output = stablehlo.add(bias, output)

  return output, mean, rstd
