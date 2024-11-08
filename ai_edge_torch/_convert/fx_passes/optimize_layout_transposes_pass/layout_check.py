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
"""Layout check for the optimized layout transposes pass."""

import dataclasses
import operator

from ai_edge_torch import lowertools
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_rewrite
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import utils
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass.op_func_registry import OpFuncRegistry
import torch
from torch.fx import Node

aten = torch.ops.aten

__all__ = [
    "is_4d",
    "can_be_nhwc",
    "must_be_nhwc",
    "get_layout_sensitive_inputs",
    "get_no_rewriter_nhwc_ops",
]


class LayoutSensitiveInputsGettersRegistry(OpFuncRegistry):

  def __missing__(self, op):

    def _default_getter(node: Node):
      """Default layout sensitive inputs are all input nodes."""
      return node.all_input_nodes

    return _default_getter


@dataclasses.dataclass
class NHWCable:
  can_be: bool
  must_be: bool

  def __bool__(self):
    raise RuntimeError(
        "Boolean value on NHWCable is disabled. Please call .can_be or .must_be"
    )


class NHWCableNodeCheckersRegistry(OpFuncRegistry):

  def __init__(self):
    self.no_rewriter_nhwc_ops = set()

  def __missing__(self, op):

    def _default_checker(node: Node):
      """Default checker for most of the layout insensitive ops.

      The node should be marked and rewritten to NHWC if:
      1. The node output is a single 4-D tensor.
      2. All layout sensitive input nodes (default all inputs) of this
        node are all marked as NHWC.
      3. All layout sensitive input nodes return 4-D tensors.
      4. There exists a rewrite rule for this node (explicit registry
        required for noop.)
      """
      nonlocal self
      layout_sensitive_inputs = get_layout_sensitive_inputs(node)

      can_be_nhwc = is_4d(node) and all_layout_sensitive_inputs_are_4d(node)
      has_rewriter = layout_rewrite.has_nhwc_rewriter(node)

      if can_be_nhwc and not has_rewriter:
        self.no_rewriter_nhwc_ops.add(node.target)

      return NHWCable(can_be_nhwc and has_rewriter, must_be=False)

    return _default_checker


nhwcable_node_checkers = NHWCableNodeCheckersRegistry()
layout_sensitive_inputs_getters = LayoutSensitiveInputsGettersRegistry()


def can_be_nhwc(node: Node):
  return nhwcable_node_checkers[node.target](node).can_be


def must_be_nhwc(node: Node):
  return nhwcable_node_checkers[node.target](node).must_be


def get_layout_sensitive_inputs(node: Node):
  return layout_sensitive_inputs_getters[node.target](node)


def get_no_rewriter_nhwc_ops():
  """Debug only: get the ops that may be NHWC but not due to no rewriter registered."""
  return nhwcable_node_checkers.no_rewriter_nhwc_ops


def is_4d(node: Node):
  val = node.meta.get("val")
  if val is None:
    return False

  if isinstance(val, (list, tuple)) and val:
    val = val[0]

  if not hasattr(val, "shape"):
    return False

  return len(val.shape) == 4


def all_layout_sensitive_inputs_are_4d(node: Node):
  return all(is_4d(m) for m in get_layout_sensitive_inputs(node))


# ==== Quantize ops (use default NHWC checker)


@layout_sensitive_inputs_getters.register(
    torch.ops.quantized_decomposed.dequantize_per_tensor
)
@layout_sensitive_inputs_getters.register(
    torch.ops.quantized_decomposed.quantize_per_tensor
)
@layout_sensitive_inputs_getters.register(
    torch.ops.quantized_decomposed.dequantize_per_channel
)
@layout_sensitive_inputs_getters.register(
    torch.ops.quantized_decomposed.quantize_per_channel
)
def _qdq_layout_sensitive_inputs_getter(node: Node):
  return [node.args[0]]


# ==== Ops must be NHWC if possible


@layout_sensitive_inputs_getters.register(aten.conv2d)
@layout_sensitive_inputs_getters.register(aten.convolution)
@layout_sensitive_inputs_getters.register(
    aten._native_batch_norm_legit_no_training
)
@layout_sensitive_inputs_getters.register(aten.native_group_norm)
def _first_arg_getter(node):
  return [node.args[0]]


# Note: default layout sensitive inputs are all inputs when not specified.
@nhwcable_node_checkers.register(aten.max_pool2d)
@nhwcable_node_checkers.register(aten.max_pool2d_with_indices)
@nhwcable_node_checkers.register(aten.amax)
@nhwcable_node_checkers.register(aten.avg_pool2d)
@nhwcable_node_checkers.register(aten._prelu_kernel)
@nhwcable_node_checkers.register(aten.upsample_bilinear2d)
@nhwcable_node_checkers.register(aten.upsample_nearest2d)
@nhwcable_node_checkers.register(aten._adaptive_avg_pool2d)
@nhwcable_node_checkers.register(aten.conv2d)
@nhwcable_node_checkers.register(aten.convolution)
def _all_layout_sensitive_inputs_are_4d_checker(node: Node):
  can_be = all_layout_sensitive_inputs_are_4d(node)
  return NHWCable(can_be, must_be=can_be)


@nhwcable_node_checkers.register(aten._native_batch_norm_legit_no_training)
def _aten_norm_checker(node):
  val = node.meta.get("val")
  if (
      not isinstance(val, (list, tuple))
      or not val
      or not hasattr(val[0], "shape")
  ):
    return NHWCable(can_be=False, must_be=False)
  return NHWCable(can_be=len(val[0].shape) == 4, must_be=False)


@nhwcable_node_checkers.register(aten.native_group_norm)
def _aten_native_group_norm_checker(node):
  val = node.meta.get("val")
  if (
      not isinstance(val, (list, tuple))
      or not val
      or not hasattr(val[0], "shape")
  ):
    return NHWCable(can_be=False, must_be=False)
  if len(node.args) >= 3 and (
      node.args[1] is not None or node.args[2] is not None
  ):
    # Disable NHWC rewriter due to precision issue with weight and bias.
    # TODO(b/354780253): Re-enable NHWC rewriter with proper lowering.
    return NHWCable(can_be=False, must_be=False)
  return NHWCable(can_be=len(val[0].shape) == 4, must_be=False)


# ==== Ops must be NCHW


@nhwcable_node_checkers.register(lowertools.mark_tensor_op)
@nhwcable_node_checkers.register(utils.tensor_to_nchw)
@nhwcable_node_checkers.register(utils.tensor_to_nhwc)
@nhwcable_node_checkers.register("output")
@nhwcable_node_checkers.register(aten.view)
@nhwcable_node_checkers.register(aten.unsqueeze_copy)
@nhwcable_node_checkers.register(aten.expand)
@nhwcable_node_checkers.register(aten.permute)
@nhwcable_node_checkers.register(aten.as_strided)
def _not_nhwc(node: Node):
  return NHWCable(can_be=False, must_be=False)


# ==== Others


@layout_sensitive_inputs_getters.register(aten.index)
@layout_sensitive_inputs_getters.register(aten._unsafe_index)
def _aten_index_layout_sensitive_inputs_getter(node):
  return [node.args[0]]


@nhwcable_node_checkers.register(aten.index)
@nhwcable_node_checkers.register(aten._unsafe_index)
def _aten_index_checker(node):
  layout_sensitive_inputs = get_layout_sensitive_inputs(node)
  can_be = is_4d(node) and all_layout_sensitive_inputs_are_4d(node)
  return NHWCable(can_be, must_be=False)


@nhwcable_node_checkers.register(operator.getitem)
def _getitem_checker(node):
  src = node.args[0]
  return nhwcable_node_checkers[src.target](src)
