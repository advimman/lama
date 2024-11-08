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
import functools
import logging

from ai_edge_torch.odml_torch import jax_bridge
import torch
import torch_xla2.ops.jaten  # Import to load torch_xla2 ops
import torch_xla2.ops.ops_registry  # Import to load torch_xla2 ops

from . import registry


@functools.cache
def _log_usage(op):
  logging.warning("Use jax lowering: %s", str(op))


def lower_by_jax(op, ir_input_names=None):
  def inner(lowering):
    bridged = jax_bridge.wrap(lowering, ir_input_names)

    @registry.lower(op)
    def _jax_lowering(lctx, *args, **kwargs):
      _log_usage(op)
      return bridged(lctx, *args, **kwargs)

    return lowering

  return inner


_TORCH_XLA2_IMPLS = {}

for op, torch_xla2_op in torch_xla2.ops.ops_registry.all_aten_ops.items():
  if not torch_xla2_op.is_jax_function:
    continue
  if isinstance(op, torch._ops.OpOverloadPacket):
    ops = [getattr(op, overload) for overload in op.overloads()] + [op]
  else:
    ops = [op]

  for op in ops:
    _TORCH_XLA2_IMPLS[op] = torch_xla2_op.func


def lower_by_torch_xla2(op):
  return lower_by_jax(op)(_TORCH_XLA2_IMPLS[op])


lower_by_torch_xla2(torch.ops.aten._adaptive_avg_pool2d)
lower_by_torch_xla2(torch.ops.aten._adaptive_avg_pool3d)
lower_by_torch_xla2(torch.ops.aten._cdist_forward)
lower_by_torch_xla2(torch.ops.aten._local_scalar_dense)
lower_by_torch_xla2(torch.ops.aten._local_scalar_dense)
lower_by_torch_xla2(torch.ops.aten._log_softmax)
lower_by_torch_xla2(torch.ops.aten._native_batch_norm_legit)
lower_by_torch_xla2(torch.ops.aten._native_batch_norm_legit_no_training)
lower_by_torch_xla2(torch.ops.aten._pdist_forward)
lower_by_torch_xla2(torch.ops.aten._softmax)
lower_by_torch_xla2(torch.ops.aten._to_copy)
lower_by_torch_xla2(torch.ops.aten._unsafe_index)
lower_by_torch_xla2(torch.ops.aten._unsafe_view)
lower_by_torch_xla2(torch.ops.aten.abs)
lower_by_torch_xla2(torch.ops.aten.acos)
lower_by_torch_xla2(torch.ops.aten.acosh)
lower_by_torch_xla2(torch.ops.aten.add.Scalar)
lower_by_torch_xla2(torch.ops.aten.add.Tensor)
lower_by_torch_xla2(torch.ops.aten.addbmm.default)
lower_by_torch_xla2(torch.ops.aten.addmm)
lower_by_torch_xla2(torch.ops.aten.addmv)
lower_by_torch_xla2(torch.ops.aten.alias)
lower_by_torch_xla2(torch.ops.aten.allclose)
lower_by_torch_xla2(torch.ops.aten.amax)
lower_by_torch_xla2(torch.ops.aten.amin)
lower_by_torch_xla2(torch.ops.aten.any)
lower_by_torch_xla2(torch.ops.aten.arange.default)
lower_by_torch_xla2(torch.ops.aten.arange.start)
lower_by_torch_xla2(torch.ops.aten.arange.start_step)
lower_by_torch_xla2(torch.ops.aten.argmax)
lower_by_torch_xla2(torch.ops.aten.argmin)
lower_by_torch_xla2(torch.ops.aten.as_strided)
lower_by_torch_xla2(torch.ops.aten.as_strided_copy)
lower_by_torch_xla2(torch.ops.aten.asin)
lower_by_torch_xla2(torch.ops.aten.asinh)
lower_by_torch_xla2(torch.ops.aten.atan)
lower_by_torch_xla2(torch.ops.aten.atan2)
lower_by_torch_xla2(torch.ops.aten.atanh)
lower_by_torch_xla2(torch.ops.aten.avg_pool2d)
lower_by_torch_xla2(torch.ops.aten.avg_pool3d)
lower_by_torch_xla2(torch.ops.aten.bitwise_and)
lower_by_torch_xla2(torch.ops.aten.bitwise_not)
lower_by_torch_xla2(torch.ops.aten.bitwise_or)
lower_by_torch_xla2(torch.ops.aten.bitwise_xor)
lower_by_torch_xla2(torch.ops.aten.bmm)
lower_by_torch_xla2(torch.ops.aten.cat)
lower_by_torch_xla2(torch.ops.aten.ceil)
lower_by_torch_xla2(torch.ops.aten.clamp.Tensor)
lower_by_torch_xla2(torch.ops.aten.clamp.default)
lower_by_torch_xla2(torch.ops.aten.clone)
lower_by_torch_xla2(torch.ops.aten.clone.default)
lower_by_torch_xla2(torch.ops.aten.constant_pad_nd)
lower_by_torch_xla2(torch.ops.aten.cos)
lower_by_torch_xla2(torch.ops.aten.cosh)
lower_by_torch_xla2(torch.ops.aten.cumsum)
lower_by_torch_xla2(torch.ops.aten.detach)
lower_by_torch_xla2(torch.ops.aten.diagonal)
lower_by_torch_xla2(torch.ops.aten.div)
lower_by_torch_xla2(torch.ops.aten.dot)
lower_by_torch_xla2(torch.ops.aten.embedding)
lower_by_torch_xla2(torch.ops.aten.empty)
lower_by_torch_xla2(torch.ops.aten.eq)
lower_by_torch_xla2(torch.ops.aten.erf)
lower_by_torch_xla2(torch.ops.aten.exp)
lower_by_torch_xla2(torch.ops.aten.expand)
lower_by_torch_xla2(torch.ops.aten.expand_copy)
lower_by_torch_xla2(torch.ops.aten.expm1)
lower_by_torch_xla2(torch.ops.aten.fill)
lower_by_torch_xla2(torch.ops.aten.flip)
lower_by_torch_xla2(torch.ops.aten.fmod)
lower_by_torch_xla2(torch.ops.aten.full)
lower_by_torch_xla2(torch.ops.aten.full_like)
lower_by_torch_xla2(torch.ops.aten.gather)
lower_by_torch_xla2(torch.ops.aten.ge)
lower_by_torch_xla2(torch.ops.aten.gelu)
lower_by_torch_xla2(torch.ops.aten.glu)
lower_by_torch_xla2(torch.ops.aten.glu.default)
lower_by_torch_xla2(torch.ops.aten.gt)
lower_by_torch_xla2(torch.ops.aten.hardtanh)
lower_by_torch_xla2(torch.ops.aten.index)
lower_by_torch_xla2(torch.ops.aten.index.Tensor)
lower_by_torch_xla2(torch.ops.aten.index_copy)
lower_by_torch_xla2(torch.ops.aten.index_put)
lower_by_torch_xla2(torch.ops.aten.index_select)
lower_by_torch_xla2(torch.ops.aten.isinf)
lower_by_torch_xla2(torch.ops.aten.isnan)
lower_by_torch_xla2(torch.ops.aten.le)
lower_by_torch_xla2(torch.ops.aten.leaky_relu)
lower_by_torch_xla2(torch.ops.aten.lift_fresh_copy)
lower_by_torch_xla2(torch.ops.aten.linalg_vector_norm)
lower_by_torch_xla2(torch.ops.aten.log)
lower_by_torch_xla2(torch.ops.aten.log10)
lower_by_torch_xla2(torch.ops.aten.log1p)
lower_by_torch_xla2(torch.ops.aten.log2)
lower_by_torch_xla2(torch.ops.aten.logical_and)
lower_by_torch_xla2(torch.ops.aten.logical_not)
lower_by_torch_xla2(torch.ops.aten.logical_or)
lower_by_torch_xla2(torch.ops.aten.logical_xor)
lower_by_torch_xla2(torch.ops.aten.lt)
lower_by_torch_xla2(torch.ops.aten.max)
lower_by_torch_xla2(torch.ops.aten.max_pool2d_with_indices)
lower_by_torch_xla2(torch.ops.aten.max_pool2d_with_indices_backward)
lower_by_torch_xla2(torch.ops.aten.max_pool2d_with_indices_backward)
lower_by_torch_xla2(torch.ops.aten.max_pool3d_with_indices)
lower_by_torch_xla2(torch.ops.aten.maximum)
lower_by_torch_xla2(torch.ops.aten.mean)
lower_by_torch_xla2(torch.ops.aten.min)
lower_by_torch_xla2(torch.ops.aten.minimum)
lower_by_torch_xla2(torch.ops.aten.mm)
lower_by_torch_xla2(torch.ops.aten.mul.Scalar)
lower_by_torch_xla2(torch.ops.aten.mul.Tensor)
lower_by_torch_xla2(torch.ops.aten.native_batch_norm)
lower_by_torch_xla2(torch.ops.aten.native_group_norm)
lower_by_torch_xla2(torch.ops.aten.native_layer_norm_backward)
lower_by_torch_xla2(torch.ops.aten.ne)
lower_by_torch_xla2(torch.ops.aten.neg)
lower_by_torch_xla2(torch.ops.aten.nonzero)
lower_by_torch_xla2(torch.ops.aten.outer)
lower_by_torch_xla2(torch.ops.aten.permute)
lower_by_torch_xla2(torch.ops.aten.permute_copy)
lower_by_torch_xla2(torch.ops.aten.pixel_shuffle)
lower_by_torch_xla2(torch.ops.aten.pow)
lower_by_torch_xla2(torch.ops.aten.prod)
lower_by_torch_xla2(torch.ops.aten.rand)
lower_by_torch_xla2(torch.ops.aten.randn)
lower_by_torch_xla2(torch.ops.aten.reciprocal)
lower_by_torch_xla2(torch.ops.aten.reflection_pad1d)
lower_by_torch_xla2(torch.ops.aten.relu)
lower_by_torch_xla2(torch.ops.aten.remainder)
lower_by_torch_xla2(torch.ops.aten.repeat)
lower_by_torch_xla2(torch.ops.aten.reshape)
lower_by_torch_xla2(torch.ops.aten.roll)
lower_by_torch_xla2(torch.ops.aten.round)
lower_by_torch_xla2(torch.ops.aten.rsqrt)
lower_by_torch_xla2(torch.ops.aten.scalar_tensor)
lower_by_torch_xla2(torch.ops.aten.scatter.src)
lower_by_torch_xla2(torch.ops.aten.scatter.value)
lower_by_torch_xla2(torch.ops.aten.scatter_add)
lower_by_torch_xla2(torch.ops.aten.scatter_reduce)
lower_by_torch_xla2(torch.ops.aten.select)
lower_by_torch_xla2(torch.ops.aten.select_copy)
lower_by_torch_xla2(torch.ops.aten.select_scatter)
lower_by_torch_xla2(torch.ops.aten.sigmoid)
lower_by_torch_xla2(torch.ops.aten.sign)
lower_by_torch_xla2(torch.ops.aten.silu)
lower_by_torch_xla2(torch.ops.aten.sin)
lower_by_torch_xla2(torch.ops.aten.sinh)
lower_by_torch_xla2(torch.ops.aten.slice)
lower_by_torch_xla2(torch.ops.aten.slice_copy)
lower_by_torch_xla2(torch.ops.aten.sort)
lower_by_torch_xla2(torch.ops.aten.split)
lower_by_torch_xla2(torch.ops.aten.split_copy)
lower_by_torch_xla2(torch.ops.aten.split_with_sizes)
lower_by_torch_xla2(torch.ops.aten.sqrt)
lower_by_torch_xla2(torch.ops.aten.squeeze)
lower_by_torch_xla2(torch.ops.aten.squeeze_copy)
lower_by_torch_xla2(torch.ops.aten.stack)
lower_by_torch_xla2(torch.ops.aten.sub.Scalar)
lower_by_torch_xla2(torch.ops.aten.sub.Tensor)
lower_by_torch_xla2(torch.ops.aten.sum)
lower_by_torch_xla2(torch.ops.aten.sym_size)
lower_by_torch_xla2(torch.ops.aten.t)
lower_by_torch_xla2(torch.ops.aten.tan)
lower_by_torch_xla2(torch.ops.aten.tanh)
lower_by_torch_xla2(torch.ops.aten.tensor_split.sections)
lower_by_torch_xla2(torch.ops.aten.tensor_split.sections)
lower_by_torch_xla2(torch.ops.aten.to.device)
lower_by_torch_xla2(torch.ops.aten.to.device)
lower_by_torch_xla2(torch.ops.aten.to.dtype)
lower_by_torch_xla2(torch.ops.aten.topk)
lower_by_torch_xla2(torch.ops.aten.transpose)
lower_by_torch_xla2(torch.ops.aten.transpose_copy)
lower_by_torch_xla2(torch.ops.aten.triu)
lower_by_torch_xla2(torch.ops.aten.true_divide)
lower_by_torch_xla2(torch.ops.aten.trunc)
lower_by_torch_xla2(torch.ops.aten.unbind_copy)
lower_by_torch_xla2(torch.ops.aten.unsqueeze)
lower_by_torch_xla2(torch.ops.aten.unsqueeze.default)
lower_by_torch_xla2(torch.ops.aten.unsqueeze_copy)
lower_by_torch_xla2(torch.ops.aten.var.correction)
lower_by_torch_xla2(torch.ops.aten.var_mean.correction)
lower_by_torch_xla2(torch.ops.aten.view)
lower_by_torch_xla2(torch.ops.aten.view_as_complex)
lower_by_torch_xla2(torch.ops.aten.view_as_real)
lower_by_torch_xla2(torch.ops.aten.view_copy)
lower_by_torch_xla2(torch.ops.aten.where.ScalarOther)
lower_by_torch_xla2(torch.ops.aten.where.ScalarSelf)
lower_by_torch_xla2(torch.ops.aten.where.self)
lower_by_torch_xla2(torch.ops.prims.broadcast_in_dim)
lower_by_torch_xla2(torch.ops.prims.var)


@lower_by_jax(torch.ops.aten.unbind)
def _aten_copy(self, *args, **kwargs):
  return _TORCH_XLA2_IMPLS[torch.ops.aten.unbind_copy](self, *args, **kwargs)


@lower_by_jax(torch.ops.aten.copy, ir_input_names=["src"])
def _aten_copy(self, src, **kwargs):
  return _TORCH_XLA2_IMPLS[torch.ops.aten.copy](self, src)
