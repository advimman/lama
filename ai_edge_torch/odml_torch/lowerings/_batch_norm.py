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
"""Provides lowering for coreaten to mlir stablehlo op: Convolution"""

from typing import Optional

from ai_edge_torch.odml_torch.lowerings import utils
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch

from .registry import lower


# _native_batch_norm_legit_no_training(
#     Tensor input,
#     Tensor? weight,
#     Tensor? bias,
#     Tensor running_mean,
#     Tensor running_var,
#     float momentum,
#     float eps) -> (Tensor, Tensor, Tensor)
@lower(torch.ops.aten._native_batch_norm_legit_no_training)
def _native_batch_norm_legit_no_training(
    lctx,
    input_tensor: ir.Value,
    weight: Optional[ir.Value],
    bias: Optional[ir.Value],
    running_mean: ir.Value,
    running_var: ir.Value,
    momentum: float,
    eps: float,
):
  if weight is None:
    weight = utils.splat(
        1, running_mean.type.element_type, running_mean.type.shape
    )
  if bias is None:
    bias = utils.splat(
        0, running_mean.type.element_type, running_mean.type.shape
    )

  return [
      stablehlo.batch_norm_inference(
          input_tensor, weight, bias, running_mean, running_var, eps, 1
      ),
      utils.splat(
          0, input_tensor.type.element_type
      ),  # TODO: return empty array instead
      utils.splat(
          0, input_tensor.type.element_type
      ),  # TODO: return empty array instead
  ]
