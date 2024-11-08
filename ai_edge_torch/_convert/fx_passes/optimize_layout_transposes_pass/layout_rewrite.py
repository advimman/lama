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
"""Layout rewrite for the optimized layout transposes pass."""

import operator

from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_mark
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import op_func_registry
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import utils
import torch
import torch.utils._pytree as pytree

aten = torch.ops.aten

__all__ = ["rewrite_nhwc_node", "has_nhwc_rewriter"]


class NHWCNodeRewritersRegistry(op_func_registry.OpFuncRegistry):

  def __missing__(self, op):
    def _rewriter(node):
      raise RuntimeError(f"NHWC node rewriter not found: {str(node)}")

    return _rewriter


rewriters = NHWCNodeRewritersRegistry()


def rewrite_nhwc_node(node: torch.fx.Node):
  if not layout_mark.is_nhwc_node(node):
    return

  rewriters[node.target](node)


def has_nhwc_rewriter(node: torch.fx.Node):
  return node.target in rewriters


# ======= Quantize ops


@rewriters.register(torch.ops.quantized_decomposed.dequantize_per_tensor)
@rewriters.register(torch.ops.quantized_decomposed.quantize_per_tensor)
def noop(node: torch.fx.Node):
  pass


@rewriters.register(torch.ops.quantized_decomposed.dequantize_per_channel)
@rewriters.register(torch.ops.quantized_decomposed.quantize_per_channel)
def _qdq_per_channel_rewriter(node: torch.fx.Node):
  new_args = []
  new_kwargs = {}

  def axis_nchw_to_nhwc(axis: int):
    axis = axis if axis >= 0 else 4 + axis
    return {3: 2, 2: 1, 1: 3}.get(axis, axis)

  for arg, spec in zip(node.args, op._schema.arguments):
    if spec.name == "axis":
      new_args.append(axis_nchw_to_nhwc(arg))
    else:
      new_args.append(arg)

  for spec in op._schema.arguments[len(node.args) :]:
    if spec.name not in node.kwargs:
      continue

    if spec.name == "axis":
      new_kwargs[spec.name] = axis_nchw_to_nhwc(node.kwargs[spec.name])
    else:
      new_kwargs[spec.name] = node.kwargs[spec.name]

  node.args = tuple(new_args)
  node.kwargs = new_kwargs


# ======= Noop ops (layout insensitive ops)


@rewriters.register(utils.tensor_to_nhwc)
@rewriters.register(utils.tensor_to_nchw)
@rewriters.register(operator.getitem)
@rewriters.register("output")
@rewriters.register(aten.add.Tensor)
@rewriters.register(aten.add.Scalar)
@rewriters.register(aten.atan2.default)
@rewriters.register(aten.atan2.out)
@rewriters.register(aten.bitwise_and.Tensor)
@rewriters.register(aten.bitwise_and.Scalar)
@rewriters.register(aten.bitwise_or.Tensor)
@rewriters.register(aten.bitwise_or.Scalar)
@rewriters.register(aten.bitwise_xor.Tensor)
@rewriters.register(aten.bitwise_xor.Scalar)
@rewriters.register(aten.div.Tensor)
@rewriters.register(aten.div.Scalar)
@rewriters.register(aten.div.Tensor_mode)
@rewriters.register(aten.div.Scalar_mode)
@rewriters.register(aten.fmod.Tensor)
@rewriters.register(aten.fmod.Scalar)
@rewriters.register(aten.mul.Tensor)
@rewriters.register(aten.mul.Scalar)
@rewriters.register(aten.remainder.Tensor)
@rewriters.register(aten.remainder.Scalar)
@rewriters.register(aten.sub.Tensor)
@rewriters.register(aten.sub.Scalar)
@rewriters.register(aten.eq.Tensor)
@rewriters.register(aten.eq.Scalar)
@rewriters.register(aten.ne.Tensor)
@rewriters.register(aten.ne.Scalar)
@rewriters.register(aten.le.Tensor)
@rewriters.register(aten.le.Scalar)
@rewriters.register(aten.ge.Tensor)
@rewriters.register(aten.ge.Scalar)
@rewriters.register(aten.gt.Tensor)
@rewriters.register(aten.gt.Scalar)
@rewriters.register(aten.lt.Tensor)
@rewriters.register(aten.lt.Scalar)
@rewriters.register(aten.maximum.default)
@rewriters.register(aten.minimum.default)
@rewriters.register(aten.mean.default)
@rewriters.register(aten.prod.default)
@rewriters.register(aten.abs.default)
@rewriters.register(aten.acos.default)
@rewriters.register(aten.acosh.default)
@rewriters.register(aten.asin.default)
@rewriters.register(aten.asinh.default)
@rewriters.register(aten.atan.default)
@rewriters.register(aten.atanh.default)
@rewriters.register(aten.bitwise_not.default)
@rewriters.register(aten.ceil.default)
@rewriters.register(aten.clamp.default)
@rewriters.register(aten.clamp.Tensor)
@rewriters.register(aten.cos.default)
@rewriters.register(aten.cosh.default)
@rewriters.register(aten.erf.default)
@rewriters.register(aten.exp.default)
@rewriters.register(aten.expm1.default)
@rewriters.register(aten.floor.default)
@rewriters.register(aten.log.default)
@rewriters.register(aten.log10.default)
@rewriters.register(aten.log1p.default)
@rewriters.register(aten.log2.default)
@rewriters.register(aten.isnan.default)
@rewriters.register(aten.neg.default)
@rewriters.register(aten.pow.Tensor_Tensor)
@rewriters.register(aten.pow.Tensor_Scalar)
@rewriters.register(aten.pow.Scalar)
@rewriters.register(aten.reciprocal.default)
@rewriters.register(aten.round.default)
@rewriters.register(aten.rsqrt.default)
@rewriters.register(aten.sigmoid.default)
@rewriters.register(aten.sign.default)
@rewriters.register(aten.sin.default)
@rewriters.register(aten.sinh.default)
@rewriters.register(aten.sqrt.default)
@rewriters.register(aten.tan.default)
@rewriters.register(aten.tanh.default)
@rewriters.register(aten.trunc.default)
@rewriters.register(aten.nonzero.default)
@rewriters.register(aten.copy.default)
@rewriters.register(aten.mm.default)
@rewriters.register(aten.fill.Scalar)
@rewriters.register(aten.col2im.default)
@rewriters.register(aten.addmm.default)
@rewriters.register(aten.gelu.default)
@rewriters.register(aten.hardtanh.default)
@rewriters.register(aten.leaky_relu.default)
@rewriters.register(aten.relu.default)
@rewriters.register(aten.arange.start_step)
@rewriters.register(aten.isinf.default)
@rewriters.register(aten.logical_and.default)
@rewriters.register(aten.logical_not.default)
@rewriters.register(aten.logical_or.default)
@rewriters.register(aten.logical_xor.default)
@rewriters.register(aten.where.self)
@rewriters.register(aten.clone.default)
@rewriters.register(aten.any.default)
@rewriters.register(aten.repeat.default)
@rewriters.register(aten.alias.default)
@rewriters.register(aten._pdist_forward.default)
@rewriters.register(aten._cdist_forward.default)
@rewriters.register(aten.bmm.default)
@rewriters.register(aten.hardswish)
@rewriters.register(aten.hardsigmoid)
@rewriters.register(aten._to_copy)
@rewriters.register(aten._prelu_kernel)
@rewriters.register(aten.softplus)
@rewriters.register(aten.silu)
def noop(node: torch.fx.Node):
  pass


# ======= Add transposes before and after NCHW-only ops (T-aten-T)


@rewriters.register(aten.upsample_bilinear2d)
@rewriters.register(aten.upsample_nearest2d)
@rewriters.register(aten.max_pool2d)
@rewriters.register(aten.max_pool2d_with_indices)
@rewriters.register(aten.avg_pool2d)
@rewriters.register(aten._adaptive_avg_pool2d.default)
def transpose_first_arg_rewriter(node: torch.fx.Node):
  op = node.target

  def nhwc_op(x, *args, **kwargs):
    nonlocal op
    x = utils.tensor_to_nchw(x)
    res = pytree.tree_map_only(
        torch.Tensor,
        utils.tensor_to_nhwc,
        op(x, *args, **kwargs),
    )
    return res

  node.target = nhwc_op


@rewriters.register(aten.conv2d)
@rewriters.register(aten.convolution)
def _aten_convolution_rewriter(node: torch.fx.Node):
  op = node.target

  def conv_nhwc(input, weight, bias=None, *args, **kwargs):
    nonlocal op
    nhwc_bias = None
    if bias is not None and len(bias.shape) == 1:
      nhwc_bias = bias
      bias = None

    input = utils.tensor_to_nchw(input)
    res = pytree.tree_map_only(
        torch.Tensor,
        utils.tensor_to_nhwc,
        op(input, weight, bias, *args, **kwargs),
    )

    if nhwc_bias is not None:
      res += nhwc_bias
    return res

  node.target = conv_nhwc


# ======= Rewrite dim attribute(s)


@rewriters.register(aten._softmax.default)
@rewriters.register(aten.select.int)
@rewriters.register(aten.slice.Tensor)
@rewriters.register(aten.sum.dim_IntList)
@rewriters.register(aten.mean.dim)
@rewriters.register(aten.prod.dim_int)
@rewriters.register(aten.var.dim)
@rewriters.register(aten.var.correction)
@rewriters.register(aten.slice_scatter.default)
@rewriters.register(aten.diagonal.default)
@rewriters.register(aten.select_scatter.default)
@rewriters.register(aten.sym_size.int)
@rewriters.register(aten.sym_stride.int)
@rewriters.register(aten._log_softmax.default)
@rewriters.register(aten.split_with_sizes.default)
@rewriters.register(aten.squeeze.dim)
@rewriters.register(aten.squeeze.dims)
@rewriters.register(aten.scatter.value)
@rewriters.register(aten.scatter.src)
@rewriters.register(aten.scatter_add.default)
@rewriters.register(aten.scatter_reduce.two)
@rewriters.register(aten.any.dim)
@rewriters.register(aten.any.dims)
@rewriters.register(aten.flip.default)
@rewriters.register(aten.index_select.default)
@rewriters.register(aten.cumsum.default)
@rewriters.register(aten.max.dim)
@rewriters.register(aten.min.dim)
@rewriters.register(aten.gather.default)
@rewriters.register(aten.sort.default)
@rewriters.register(aten.topk.default)
@rewriters.register(aten.cat.default)
def dim_attr_rewriter(node: torch.fx.Node):
  op = node.target

  new_args = []
  new_kwargs = {}

  def dims_nchw_to_nhwc(dims: list[int]):
    def convert(dim: int):
      dim = dim if dim >= 0 else 4 + dim
      return {3: 2, 2: 1, 1: 3}.get(dim, dim)

    dims = pytree.tree_map_only(int, convert, dims)
    dims = pytree.tree_map_only(torch.SymInt, convert, dims)
    return dims

  for arg, spec in zip(node.args, op._schema.arguments):
    if spec.name.startswith("dim"):
      new_args.append(dims_nchw_to_nhwc(arg))
    else:
      new_args.append(arg)

  for spec in op._schema.arguments[len(node.args) :]:
    if spec.name not in node.kwargs:
      continue

    if spec.name.startswith("dim"):
      new_kwargs[spec.name] = dims_nchw_to_nhwc(node.kwargs[spec.name])
    else:
      new_kwargs[spec.name] = node.kwargs[spec.name]

  node.args = tuple(new_args)
  node.kwargs = new_kwargs


# ======= Others


@rewriters.register(aten._native_batch_norm_legit_no_training.default)
def _aten__native_batch_norm_legit_no_training(node):
  def batch_norm(input, weight, bias, running_mean, running_var, momentum, eps):
    a = input - running_mean
    b = torch.sqrt(running_var + eps)
    out = a / b
    if weight is not None:
      out = out * weight
    if bias is not None:
      out = out + bias
    return out, None, None

  node.target = batch_norm


@rewriters.register(aten.native_group_norm.default)
def _aten_native_group_norm(node):

  def native_group_norm(
      input,
      weight,
      bias,
      batch_size: int,
      num_channels: int,
      flattened_inner_size: int,
      num_groups: int,
      eps: float,
  ):
    input_reshaped = torch.reshape(
        input,
        [
            batch_size,
            flattened_inner_size,
            num_groups,
            num_channels // num_groups,
        ],
    )
    reduction_dims = [1, 3]

    biased_var, mean = torch.var_mean(
        input_reshaped, dim=reduction_dims, unbiased=False, keepdim=True
    )
    rstd = torch.rsqrt(biased_var + eps)

    out = (input_reshaped - mean) * rstd
    out = torch.reshape(out, input.shape)

    if weight is not None:
      out = out * weight
    if bias is not None:
      out = out + bias

    mean = torch.squeeze(mean, reduction_dims)
    rstd = torch.squeeze(rstd, reduction_dims)

    return out, mean, rstd

  node.target = native_group_norm


@rewriters.register(aten.index)
@rewriters.register(aten._unsafe_index)
def _aten_index(node):
  op = node.target

  def index_nhwc(x, indices=[], *args, **kwargs):
    nonlocal op
    indices = list(indices)
    if len(indices) < 4:
      indices += [None] * (4 - len(indices))

    indices[1:4] = indices[2], indices[3], indices[1]
    return op(x, indices, *args, **kwargs)

  node.target = index_nhwc


@rewriters.register(aten.reflection_pad2d.default)
def _aten_reflection_pad2d(node):
  def reflection_pad2d_nhwc(x, padding):
    padding = [0, 0] + padding
    return torch.nn.functional.pad(x, padding, mode="reflect")

  node.target = reflection_pad2d_nhwc
