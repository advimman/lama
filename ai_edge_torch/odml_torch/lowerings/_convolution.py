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
"""Provides lowering for coreaten to stablehlo for Convolution."""

import math
from typing import Optional

from ai_edge_torch.odml_torch.lowerings import registry
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch


def make_padding(padding):
  """Change the padding from pytorch to stablehlo style.

  Stablehlo allows start and end padding for each dimension while aten only
  allows symmetric padding and so only has one number per dimension.

  Args:
    padding: The padding of the convolution

  Returns:
    The padding in stablehlo style
  """
  return tuple((p, p) for p in padding)


def create_conv_dimension_numbers(lhs, transposed: bool = False):
  """Create the dimension numbers for the convolution.

  Args:
    lhs: The input tensor
    transposed: Whether the convolution is transposed

  Returns:
    The dimension numbers for the convolution
  """
  num_spatial_dims = len(lhs.type.shape) - 2
  spatial_dimensions = []
  for i in range(0, num_spatial_dims):
    spatial_dimensions.append(i + 2)

  # Regular kernels are OIHW
  # TransposedConv kernels are IOHW
  dimension_numbers = stablehlo.ConvDimensionNumbers.get(
      input_batch_dimension=0,
      input_feature_dimension=1,
      input_spatial_dimensions=spatial_dimensions,
      kernel_input_feature_dimension=0 if transposed else 1,
      kernel_output_feature_dimension=1 if transposed else 0,
      kernel_spatial_dimensions=spatial_dimensions,
      output_batch_dimension=0,
      output_feature_dimension=1,
      output_spatial_dimensions=spatial_dimensions,
  )
  return dimension_numbers


def infer_output_shape(
    lhs,
    rhs,
    stride,
    dilation,
    padding,
    transposed: bool = False,
    output_padding: list[int] = 0,
):
  """Infer the output shape of the convolution.

  Args:
    lhs: The input tensor
    rhs: The kernel tensor
    stride: The stride of the convolution (dilation of input in transposed conv)
    dilation: The kernel dilation of the convolution
    padding: The padding of the convolution
    transposed: Whether the convolution is transposed
    output_padding: The output padding of the convolution

  Returns:
    The output shape of the convolution
  """
  lhs_type: ir.RankedTensorType = lhs.type
  lhs_shape: list[int] = lhs_type.shape
  rhs_shape: list[int] = rhs.type.shape

  # Input layout is: (N)CHW and Kernel layout is: (O)IHW for regular conv
  # Input layout is: (N)CHW and Kernel layout is: I(O)HW for transposed conv
  output_shape = (
      [lhs_shape[0], rhs_shape[1]]
      if transposed
      else [lhs_shape[0], rhs_shape[0]]
  )
  num_spatial_dims = len(lhs.type.shape) - 2

  # looping over the spatial dims (skipping the first 2 dims which are
  # batch and features)
  for spatial_dim in range(0, num_spatial_dims):
    dim = spatial_dim + 2
    dim_size = lhs_shape[dim]
    kernel_dim_size = rhs_shape[dim]

    if transposed:
      output_dim_size = (
          (dim_size - 1) * stride[spatial_dim]
          - 2 * padding[spatial_dim]
          + dilation[spatial_dim] * (kernel_dim_size - 1)
          + output_padding[spatial_dim]
          + 1
      )
    else:
      output_dim_size = math.floor(
          (
              (
                  dim_size
                  + 2 * padding[spatial_dim]
                  - dilation[spatial_dim] * (kernel_dim_size - 1)
                  - 1
              )
              / stride[spatial_dim]
          )
          + 1
      )

    output_shape.append(output_dim_size)

  return output_shape


def build_transpose_conv(
    lctx,
    output_type: ir.RankedTensorType,
    lhs: ir.Value,
    rhs: ir.Value,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    output_padding: list[int],
    groups: int,
):
  lhs_type: ir.RankedTensorType = lhs.type
  num_spatial_dims = len(lhs_type.shape) - 2
  rhs = stablehlo.reverse(rhs, list(range(2, 2 + num_spatial_dims)))

  kernel_size = rhs.type.shape
  # We need to additional padding on the input to get the right output size.
  adjusted_padding = [
      dilation[dim] * (kernel_size[dim + 2] - 1) - padding[dim]
      for dim in range(num_spatial_dims)
  ]
  return stablehlo.convolution(
      result=output_type,
      lhs=lhs,
      rhs=rhs,
      dimension_numbers=create_conv_dimension_numbers(lhs, True),
      feature_group_count=groups,
      batch_group_count=1,
      padding=make_padding(adjusted_padding),
      lhs_dilation=stride,
      rhs_dilation=dilation,
  )


# convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride,
#   SymInt[] padding, SymInt[] dilation, bool transposed,
#   SymInt[] output_padding, SymInt groups) -> Tensor
@registry.lower(torch.ops.aten.convolution)
def _aten_convolution(
    lctx,
    lhs: ir.Value,
    rhs: ir.Value,
    bias: Optional[ir.Value],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
):

  # TODO(b/365559296) Add support for output_padding
  if any(output_padding):
    raise NotImplementedError(
        "Output padding on convolution is not implemented."
    )

  lhs_type: ir.RankedTensorType = lhs.type
  output_shape = infer_output_shape(
      lhs, rhs, stride, dilation, padding, transposed, output_padding
  )
  output_type = ir.RankedTensorType.get(
      output_shape,
      lhs_type.element_type,
  )

  if transposed:
    res = build_transpose_conv(
        lctx,
        output_type,
        lhs,
        rhs,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
    )
  else:
    res = stablehlo.convolution(
        result=output_type,
        lhs=lhs,
        rhs=rhs,
        dimension_numbers=create_conv_dimension_numbers(lhs),
        feature_group_count=groups,
        batch_group_count=1,
        window_strides=stride,
        padding=make_padding(padding),
        rhs_dilation=dilation,
    )

  if bias is not None:
    # broadcast [C] to [NCHW]
    broadcasted_bias = stablehlo.broadcast_in_dim(output_type, bias, [1])
    res = stablehlo.add(
        lhs=res,
        rhs=broadcasted_bias,
    )

  return res
