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
"""Optimize layout transposes pass."""

import operator
import os
from typing import Union

from ai_edge_torch import fx_pass_base
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_check  # NOQA
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_mark  # NOQA
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_partitioners  # NOQA
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_rewrite  # NOQA
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import utils  # NOQA
import torch
import torch.ao.quantization.quantize_pt2e

TransposeFunc = Union[utils.tensor_to_nchw, utils.tensor_to_nhwc]


class OptimizeLayoutTransposesPass(fx_pass_base.ExportedProgramPassBase):

  def get_source_meta(self, node: torch.fx.Node):
    keys = ["stack_trace", "nn_module_stack", "source_fn_stack", "from_node"]
    meta = {}
    for key in keys:
      if key in node.meta:
        meta[key] = node.meta[key]
    return meta

  def insert_t_q_dq(
      self,
      graph: torch.fx.Graph,
      input_dq: torch.fx.Node,
      target: torch.fx.Node,
      transpose_func: TransposeFunc,
      transpose_node_meta: dict,
  ) -> list[torch.fx.Node]:
    """original:

        input_dq -> target
    insert the node as:
        input_dq -> (T q dq) -> target
    """
    assert utils.is_dq_node(input_dq)

    q_args = input_dq.args[1:]
    q_kwargs = input_dq.kwargs
    q_op, dq_op = utils.get_paired_q_dq_ops(input_dq.target)
    with graph.inserting_before(target):
      t = graph.call_function(transpose_func, (input_dq,))
      # Q and DQ inserted here may required updating the `axis` arg when they
      # are per_channel ops. However, instead of updating here, the nodes would
      # be marked as NHWC/NCHW and applied rewriters after partitioning.
      q = graph.call_function(q_op, (t,) + q_args, q_kwargs)
      dq = graph.call_function(dq_op, (q,) + q_args, q_kwargs)

    input_dq.meta = transpose_node_meta
    t.meta = transpose_node_meta
    q.meta = transpose_node_meta
    dq.meta = self.get_source_meta(target)

    target.replace_input_with(input_dq, dq)
    return [t, q, dq]

  def insert_dq_t_q(
      self,
      graph: torch.fx.Graph,
      input_q: torch.fx.Node,
      target: torch.fx.Node,
      transpose_func: TransposeFunc,
      transpose_node_meta: dict,
  ) -> list[torch.fx.Node]:
    """original:

        input_q -> target
    insert the node as:
        input_q -> (dq T q) -> target
    """
    assert utils.is_q_node(input_q)

    q_args = input_q.args[1:]
    q_kwargs = input_q.kwargs
    q_op, dq_op = utils.get_paired_q_dq_ops(input_q.target)
    with graph.inserting_before(target):
      # Q and DQ inserted here may required updating the `axis` arg when they
      # are per_channel ops. However, instead of updating here, the nodes would
      # be marked as NHWC/NCHW and applied rewriters after partitioning.
      dq = graph.call_function(dq_op, (input_q,) + q_args, q_kwargs)
      t = graph.call_function(transpose_func, (dq,))
      q = graph.call_function(q_op, (t,) + q_args, q_kwargs)

    dq.meta = transpose_node_meta
    t.meta = transpose_node_meta
    q.meta = transpose_node_meta

    target.replace_input_with(input_q, q)
    return [dq, t, q]

  def insert_layout_transpose(
      self,
      graph: torch.fx.Graph,
      input_node: torch.fx.Node,
      target_node: torch.fx.Node,
      transpose_func: TransposeFunc,
      transpose_node_meta: dict,
  ) -> None:
    assert transpose_func in (utils.tensor_to_nchw, utils.tensor_to_nhwc)

    # new_nodes only contains Q/DQ/Transpose nodes, which are all SISO.
    # Insertion order input nodes -> output nodes
    new_nodes = []

    # Constraint Q2: the NHWC partition's entry and exit must not be output
    # edges of Q/DQ ops that are connected to a constant/weight tensor.
    while layout_mark.is_const_node(input_node) and (
        utils.is_dq_node(input_node) or utils.is_q_node(input_node)
    ):
      with graph.inserting_before(target_node):
        new_input_node = graph.node_copy(input_node)

      target_node.replace_input_with(input_node, new_input_node)

      new_nodes = [new_input_node] + new_nodes
      input_node, target_node = new_input_node.args[0], new_input_node

    if utils.is_q_node(input_node):
      # Constraint Q3: when the entry and exit is right after a q op (occur after a (dq-op-q)
      # triplet), the transpose must be added as a quantized transpose in (dq-T-q)
      # input_q -> (dq T q) -> target
      new_nodes = (
          self.insert_dq_t_q(
              graph,
              input_node,
              target_node,
              transpose_func,
              transpose_node_meta,
          )
          + new_nodes
      )
    elif utils.is_dq_node(input_node):
      # Constraint Q1: the NHWC partition's entry and exit cannot be edges
      # within (dq-op-q) triplet.
      # input_dq -> (T q dq) -> target
      new_nodes = (
          self.insert_t_q_dq(
              graph,
              input_node,
              target_node,
              transpose_func,
              transpose_node_meta,
          )
          + new_nodes
      )
    else:
      # input -> target
      with graph.inserting_before(target_node):
        t = graph.call_function(transpose_func, (input_node,))
      t.meta = transpose_node_meta
      target_node.replace_input_with(input_node, t)
      new_nodes = [t] + new_nodes

    # Mark new nodes as NCHW or NHWC
    # For all nodes before the transpose, mark it as input_marker
    # For all nodes after the transpose (incl. transpose), mark it as output_marker
    if transpose_func == utils.tensor_to_nchw:
      input_marker, target_marker = (
          layout_mark.mark_as_nhwc_node,
          layout_mark.mark_as_nchw_node,
      )
    else:
      input_marker, target_marker = (
          layout_mark.mark_as_nchw_node,
          layout_mark.mark_as_nhwc_node,
      )

    marker = input_marker
    for node in new_nodes:
      if node.target == transpose_func:
        marker = target_marker
      marker(node)
    assert marker == target_marker

  def input_to_nhwc(
      self,
      graph: torch.fx.Graph,
      input_node: torch.fx.Node,
      target_node: torch.fx.Node,
  ) -> None:
    if layout_mark.is_nhwc_node(input_node):
      return

    if not layout_check.is_4d(input_node):
      raise AssertionError(
          "Attempting to convert non-NHWC compatible node to NHWC:"
          f" {input_node}"
      )

    # Assign target node's source meta to the to_NHWC node, because the transpose
    # is added for the existence of target node.
    self.insert_layout_transpose(
        graph,
        input_node,
        target_node,
        utils.tensor_to_nhwc,
        self.get_source_meta(target_node),
    )

  def input_to_nchw(
      self,
      graph: torch.fx.Graph,
      input_node: torch.fx.Node,
      target_node: torch.fx.Node,
  ) -> None:
    if layout_mark.is_nchw_node(input_node):
      return

    self.insert_layout_transpose(
        graph,
        input_node,
        target_node,
        utils.tensor_to_nchw,
        self.get_source_meta(input_node),
    )

  def mark_const_nodes(self, exported_program: torch.export.ExportedProgram):
    graph_module = exported_program.graph_module
    graph = graph_module.graph

    input_specs = exported_program.graph_signature.input_specs
    non_user_input_names = set()
    for spec in input_specs:
      if spec.kind != torch.export.graph_signature.InputKind.USER_INPUT:
        non_user_input_names.add(spec.arg.name)

    for node in graph.nodes:
      has_input_nodes = len(node.all_input_nodes) > 0
      all_inputs_are_const = all(
          map(layout_mark.is_const_node, node.all_input_nodes)
      )
      if (
          node.name in non_user_input_names
          or (has_input_nodes and all_inputs_are_const)
          or (node.op != "placeholder" and not has_input_nodes)
      ):
        layout_mark.mark_as_const_node(node)

  def call(self, exported_program: torch.export.ExportedProgram):
    self.mark_const_nodes(exported_program)

    graph_module = exported_program.graph_module
    partitioner = os.environ.get(
        "AIEDGETORCH_LAYOUT_OPTIMIZE_PARTITIONER", None
    )
    if partitioner == "MINCUT":
      graph_module = layout_partitioners.min_cut.partition(graph_module)
    elif partitioner == "GREEDY":
      graph_module = layout_partitioners.greedy.partition(graph_module)
    else:
      # By default use min cut partitioner if possible
      if layout_partitioners.min_cut.can_partition(graph_module):
        graph_module = layout_partitioners.min_cut.partition(graph_module)
      else:
        graph_module = layout_partitioners.greedy.partition(graph_module)

    graph = graph_module.graph
    for node in list(graph.nodes):
      if node.target == operator.getitem:
        # force the layout mark of a getitem node to follow its producer.
        if layout_mark.is_nchw_node(node.args[0]):
          layout_mark.mark_as_nchw_node(node)
        else:
          layout_mark.mark_as_nhwc_node(node)

    for node in list(graph.nodes):
      if layout_mark.is_nhwc_node(node):
        for input_node in layout_check.get_layout_sensitive_inputs(node):
          self.input_to_nhwc(graph, input_node, node)
        layout_rewrite.rewrite_nhwc_node(node)
      else:
        for input_node in layout_check.get_layout_sensitive_inputs(node):
          # Note: for non-4D tensors input_to_nchw is always noop.
          self.input_to_nchw(graph, input_node, node)

    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()
    graph_module.graph.lint()
    # Mark const node again for debugging
    self.mark_const_nodes(exported_program)

    return fx_pass_base.ExportedProgramPassResult(exported_program, True)
