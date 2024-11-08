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
"""Torch op decompositions and MLIR lowerings registry."""

from typing import Any, Callable

import torch

from . import context


class LoweringRegistry:
  """Registry object for torch op decompositions and to-MLIR lowerings."""

  def __init__(self):
    self.registered_ops = {}
    self.decompositions = {}

  def lookup(self, op_or_name):
    candidate = self._get_lowering(op_or_name)
    if candidate is None:
      if isinstance(op_or_name, torch._ops.OpOverloadPacket):
        candidate = self._get_lowering(op_or_name.default)
      if isinstance(op_or_name, torch._ops.OpOverload):
        candidate = self._get_lowering(op_or_name.overloadpacket)
    return candidate

  def _get_lowering(self, op):
    candidate = self.registered_ops.get(op)
    return candidate

  def register(self, op, lowering):
    if isinstance(op, torch._ops.OpOverloadPacket):
      ops = [getattr(op, overload) for overload in op.overloads()]
    else:
      ops = [op]

    for op in ops:
      self.registered_ops[op] = lowering


global_registry = LoweringRegistry()
global_registry.decompositions.update(torch._decomp.core_aten_decompositions())
global_registry.decompositions.update(
    torch._decomp.get_decompositions([
        torch.ops.aten.upsample_nearest2d,
        torch.ops.aten._native_batch_norm_legit.no_stats,
        torch.ops.aten._native_batch_norm_legit_functional,
        torch.ops.aten._adaptive_avg_pool2d,
        torch.ops.aten._adaptive_avg_pool3d,
        torch.ops.aten.grid_sampler_2d,
        torch.ops.aten.native_dropout,
        torch.ops.aten.reflection_pad1d,
        torch.ops.aten.reflection_pad2d,
        torch.ops.aten.reflection_pad3d,
        torch.ops.aten.replication_pad1d,
        torch.ops.aten.replication_pad2d,
        torch.ops.aten.replication_pad3d,
        torch.ops.aten.addmm,
    ])
)

torch._decomp.remove_decompositions(
    global_registry.decompositions,
    [
        torch.ops.aten.roll,
    ],
)


def lookup(op):
  return global_registry.lookup(op)


def lower(op):
  def inner(lowering: Callable[[context.LoweringContext, ...], Any]):
    global_registry.register(op, lowering)
    return lowering

  return inner


def decompositions():
  return global_registry.decompositions
