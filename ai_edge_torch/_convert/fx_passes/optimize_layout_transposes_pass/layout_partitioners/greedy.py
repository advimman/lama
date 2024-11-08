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
"""Greedy partitioning algorithm."""

from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_check
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_mark
import torch


def partition(graph_module: torch.fx.GraphModule):
  """Partition the graph module into NHWC and non-NHWC subgraphs.

  Partition the graph module into NHWC and non-NHWC subgraphs and mark nodes in
  the NHWC partitions.

  Implements O(|V|) greedy partitioning algorithm.

  Args:
    graph_module: The graph module to be partitioned.

  Returns:
    The partitioned graph module.
  """
  graph = graph_module.graph

  for node in list(graph.nodes):
    if not node.all_input_nodes:
      # This node has no inputs so we don't need to change anything
      continue

    if layout_check.must_be_nhwc(node):
      # If the node has must_be_nhwc equals true, mark this node as NHWC

      layout_mark.mark_as_nhwc_node(node)
    elif layout_check.can_be_nhwc(node):
      # If the following conditions are all true, mark this node as NHWC
      # - The node has can_be_nhwc equals true
      # - Any of the node's layout sensitive inputs is marked as NHWC
      # - All the node's layout sensitive inputs are 4D tensors

      layout_sensitive_inputs = layout_check.get_layout_sensitive_inputs(node)

      should_be_nhwc = any(
          map(layout_mark.is_nhwc_node, layout_sensitive_inputs)
      )
      for input_node in layout_sensitive_inputs:
        if not layout_mark.is_nhwc_node(input_node) and not layout_check.is_4d(
            input_node
        ):
          should_be_nhwc = False

      if should_be_nhwc:
        layout_mark.mark_as_nhwc_node(node)

  graph_module.recompile()
  return graph_module
