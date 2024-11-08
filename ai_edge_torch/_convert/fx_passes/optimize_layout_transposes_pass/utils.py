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
"""Utils for the optimized layout transposes pass."""

from typing import Callable

import torch
import torch.ao.quantization.quantize_pt2e


def tensor_to_nhwc(t: torch.Tensor):
  return torch.ops.aten.permute(t.contiguous(), [0, 2, 3, 1]).contiguous()


def tensor_to_nchw(t: torch.Tensor):
  return torch.ops.aten.permute(t.contiguous(), [0, 3, 1, 2]).contiguous()


def flatten_torch_op_overloads(op):
  if isinstance(op, torch._ops.OpOverloadPacket):
    return [getattr(op, overload) for overload in op.overloads()]
  return [op]


_TORCH_Q_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor2,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]

_TORCH_DQ_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor2,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


def is_q_node(node: torch.fx.Node):
  return node.target in _TORCH_Q_OPS


def is_dq_node(node: torch.fx.Node):
  return node.target in _TORCH_DQ_OPS


def get_paired_q_dq_ops(op: Callable) -> tuple[Callable, Callable]:
  for q, dq in zip(_TORCH_Q_OPS, _TORCH_DQ_OPS):
    if op in (q, dq):
      return q, dq
  raise AssertionError(f"{op} is not a Q/DQ op.")
