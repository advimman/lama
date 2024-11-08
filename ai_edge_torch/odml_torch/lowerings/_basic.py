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
import math
from typing import Optional, Union

from ai_edge_torch.odml_torch.lowerings import utils
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import numpy as np
import torch

from .registry import lower


# add(Tensor self, Tensor other) -> Tensor
# @lower(torch.ops.aten.add)
def _aten_add(lctx, x: ir.Value, y: ir.Value, alpha=1):
  x, y = utils.upcast_to_same_type(x, y)
  x, y = utils.broadcast_args_if_needed(x, y)
  if alpha == 1:
    return stablehlo.add(x, y)

  alpha_splat = utils.splat(alpha, y.type.element_type, y.type.shape)
  return stablehlo.add(x, stablehlo.multiply(y, alpha_splat))


# mul.Tensor(Tensor self, Tensor other) -> Tensor
# @lower(torch.ops.aten.mul.Tensor)
def _aten_mul_tensor(lctx, self: ir.Value, other: ir.Value):
  self, other = utils.upcast_to_same_type(self, other)
  self, other = utils.broadcast_args_if_needed(self, other)

  return stablehlo.multiply(self, other)


# cat(Tensor[] tensors, int dim=0) -> Tensor
# @lower(torch.ops.aten.cat)
def _aten_cat(lctx, tensors: list[ir.Value], dim: int = 1):
  return stablehlo.ConcatenateOp(tensors, dim).result


# view(Tensor(a) self, SymInt[] size) -> Tensor(a)
# @lower(torch.ops.aten.view)
def _aten_view(lctx, self: ir.Value, size: list[int]):
  return stablehlo.ReshapeOp(
      ir.RankedTensorType.get(size, self.type.element_type), self
  ).result


# hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
@lower(torch.ops.aten.hardtanh)
def _aten_hardtanh(
    lctx,
    self: ir.Value,
    min_val: Union[int, float] = -1.0,
    max_val: Union[int, float] = 1.0,
):
  elty = self.type.element_type
  min_val = utils.splat(min_val, elty)
  max_val = utils.splat(max_val, elty)

  return stablehlo.clamp(min_val, self, max_val)


# mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
# mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *,
#   ScalarType? dtype=None) -> Tensor
@lower(torch.ops.aten.mean)
@lower(torch.ops.aten.mean.dim)
def _aten_mean_dim(
    lctx,
    self: ir.Value,
    dim: Optional[list[int]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
):
  self_shape = self.type.shape
  self_elty = self.type.element_type
  if dim is None:
    dim = list(range(len(self_shape)))
  dim = [len(self_shape) + d if d < 0 else d for d in dim]
  dim_ = ir.DenseI64ArrayAttr.get(np.asarray(dim, np.int64))
  dim_to_keep = [d for d in range(len(self_shape)) if d not in dim]
  dim_to_keep_ = ir.DenseI64ArrayAttr.get(np.asarray(dim_to_keep, np.int64))

  zero_ = utils.splat(0.0, self_elty)

  reduce_result_shape = [
      s for d, s in enumerate(self_shape) if d in dim_to_keep
  ]
  reduce_result_ty = ir.RankedTensorType.get(reduce_result_shape, self_elty)
  reduce_op = stablehlo.ReduceOp([reduce_result_ty], [self], [zero_], dim_)

  reducer_arg_ty = ir.RankedTensorType.get(tuple(), self_elty)
  reducer = reduce_op.regions[0].blocks.append(reducer_arg_ty, reducer_arg_ty)
  with ir.InsertionPoint(reducer):
    stablehlo.return_(
        [stablehlo.add(reducer.arguments[0], reducer.arguments[1])]
    )

  sum_ = reduce_op.result
  if keepdim:
    sum_ = stablehlo.broadcast_in_dim(
        ir.RankedTensorType.get(
            [s if d in dim_to_keep else 1 for d, s in enumerate(self_shape)],
            self_elty,
        ),
        sum_,
        dim_to_keep_,
    )

  dim_els = math.prod([s for d, s in enumerate(self_shape) if d in dim])
  dim_els_ = utils.splat(dim_els, self_elty)
  div_ = stablehlo.broadcast_in_dim(
      sum_.type, dim_els_, ir.DenseI64ArrayAttr.get([])
  )
  mean_ = stablehlo.divide(sum_, div_)

  return mean_


# https://pytorch.org/docs/stable/generated/torch.clone.html
# https://github.com/pytorch/pytorch/blob/a95ceb51a23ae33c00b3a99224143c609b1b3eb3/aten/src/ATen/native/TensorFactories.cpp#L1730
@lower(torch.ops.aten.clone)
def _aten_clone(lctx, x: ir.Value, *, memory_format=None):
  return x


# https://pytorch.org/docs/stable/generated/torch.permute.html
# https://github.com/pytorch/pytorch/blob/519151a062a9bd4f0d32a9c7c7eae47d7ed847b2/aten/src/ATen/native/TensorShape.cpp#L1448
# https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose
@lower(torch.ops.aten.permute)
def _aten_permute(lctx, x: ir.Value, dims: list[int]):
  dim = len(x.type.shape)
  return stablehlo.transpose(x, ir.DenseI64ArrayAttr.get(dims))


# https://pytorch.org/docs/stable/generated/torch.mm.html
# https://github.com/pytorch/pytorch/blob/ffabb25c489df1dc631a577c12a0c843c8b202f3/aten/src/ATen/native/LinearAlgebra.cpp#L193
# https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general
@lower(torch.ops.aten.mm)
def _aten_mm(mod, mat1: ir.Value, mat2: ir.Value) -> ir.Value:
  mat1_shape = mat1.type.shape
  mat2_shape = mat2.type.shape
  mat1_dims = len(mat1_shape)
  mat2_dims = len(mat2_shape)

  if mat1_dims != 2 or mat1_dims != 2:
    raise ValueError(
        "Both arguments must be 2D matrices, received dimensions %d and %d"
        % (mat1_dims, mat2_dims)
    )

  if mat1_shape[1] != mat2_shape[0]:
    raise ValueError(
        "mat1 and mat2 shapes cannot be multiplied, received shapes %s and %s"
        % (mat1_shape, mat2_shape)
    )

  dot_dnums = stablehlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=[],
      rhs_batching_dimensions=[],
      lhs_contracting_dimensions=(1,),
      rhs_contracting_dimensions=(0,),
  )
  return stablehlo.dot_general(
      ir.RankedTensorType.get(
          (mat1.type.shape[0], mat2.type.shape[1]), mat1.type.element_type
      ),
      mat1,
      mat2,
      dot_dnums,
  )


# https://pytorch.org/docs/stable/generated/torch.div.html
# https://openxla.org/stablehlo/spec#divide
# TODO: support rounding mode and type promotion (see torch.div spec).
# @lower(torch.ops.aten.div)
def _aten_div(mod, x, y, *, rounding_mode=None, out=None) -> ir.Value:
  # By default, PyTorch performs a "true" division like Python 3. This requires
  # casting integer input types to float to achieve the same semantics using
  # stablehlo.divide.
  if isinstance(x.type.element_type, ir.IntegerType):
    x = utils.convert_int_to_float(x)
  if isinstance(y.type.element_type, ir.IntegerType):
    y = utils.convert_int_to_float(y)

  x, y = utils.broadcast_args_if_needed(x, y)

  return stablehlo.divide(x, y)


# https://pytorch.org/docs/stable/generated/torch.floor.html
# https://openxla.org/stablehlo/spec#floor
@lower(torch.ops.aten.floor)
def _aten_floor(lctx, x: ir.Value, *, out=None) -> ir.Value:
  return stablehlo.floor(x)


# Schema:
#   - aten::slice_scatter(Tensor self, Tensor src, int dim=0, SymInt?
#       start=None, SymInt? end=None, SymInt step=1) -> Tensor
# Torch Reference:
#   - https://pytorch.org/docs/stable/generated/torch.slice_scatter.html
#   - https://github.com/pytorch/pytorch/blob/18f9331e5deb4c02ae5c206e133a9b4add49bd97/aten/src/ATen/native/TensorShape.cpp#L4002
@lower(torch.ops.aten.slice_scatter)
def _aten_slice_scatter(lctx, self, src, dim=0, start=None, end=None, step=1):
  start = start if start is not None else 0
  end = end if end is not None else self.type.shape[dim]

  start, end = np.clip(
      [start, end], -self.type.shape[dim], self.type.shape[dim]
  )

  if start < 0:
    start = self.type.shape[dim] + start
  if end < 0:
    end = self.type.shape[dim] + end

  if end <= start or np.prod(src.type.shape) == 0:
    return self

  end = start + step * math.ceil((end - start) / step) - (step - 1)
  padding_low = start
  padding_high = self.type.shape[dim] - end
  interior_padding = step - 1

  rank = len(self.type.shape)
  src = stablehlo.pad(
      src,
      utils.splat(0, src.type.element_type, []),
      edge_padding_low=[padding_low if i == dim else 0 for i in range(rank)],
      edge_padding_high=[padding_high if i == dim else 0 for i in range(rank)],
      interior_padding=[
          interior_padding if i == dim else 0 for i in range(rank)
      ],
  )
  pred = np.ones(self.type.shape, dtype=np.bool_)
  pred[*[
      slice(start, end, step) if i == dim else slice(None, None, None)
      for i in range(rank)
  ]] = False
  pred = stablehlo.constant(
      ir.DenseElementsAttr.get(
          np.packbits(pred, bitorder="little"),
          type=ir.IntegerType.get_signless(1),
          shape=pred.shape,
      )
  )
  out = stablehlo.select(pred, self, src)
  return out
