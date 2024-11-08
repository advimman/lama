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
import copy
from typing import Any
import uuid

from ai_edge_torch import lowertools
from ai_edge_torch.hlfb.mark_pattern import passes
from ai_edge_torch.hlfb.mark_pattern import pattern as pattern_module
import torch


@torch._dynamo.assume_constant_result
def _get_uuid() -> str:
  return uuid.uuid4().hex


# TODO: Move to a general fx utils file.
def _prepose_placeholder_nodes(graph: torch.fx.Graph):
  nodes = [node for node in graph.nodes if node.op == "placeholder"] + [
      node for node in graph.nodes if node.op != "placeholder"
  ]

  for a, b in zip(nodes, nodes[1:]):
    if a.next is not b:
      a.append(b)
  return graph


def _insert_marker(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    name: str,
    pos: int,
    id: str,
    is_input: bool,
    attr: dict[str, Any] = None,
):
  attr = lowertools.serialize_composite_attr(attr) if attr else None
  with graph_module.graph.inserting_after(node):
    new_node = graph_module.graph.call_function(
        lowertools.mark_tensor_op,
        args=(node,),
        kwargs={
            "name": name,
            "pos": pos,
            "id": id,
            "is_input": is_input,
            "attr": attr,
        },
    )

  new_node.meta = node.meta
  return new_node


def mark_pattern(
    graph_module: torch.fx.GraphModule,
    pattern: pattern_module.Pattern,
) -> torch.fx.GraphModule:
  """Mark all existences of pattern graph in the GraphModule with fx pattern matching.

  The marked subgraphs will be lowered in StableHLO composite ops.

  Args:
    graph_module (torch.fx.GraphModule): GraphModule to be matched and marked.
    pattern (ai_edge_torch.hlfb.mark_pattern.Pattern): Pattern to match.

  Returns:
    The modified graph_module with additional marker ops in graph.
  """
  # Create a copy of graph_module and sanitize it for pattern matching.
  graph_module_to_match = copy.deepcopy(graph_module)
  for n, m in zip(graph_module.graph.nodes, graph_module_to_match.graph.nodes):
    m.meta["ORIGINAL_NODE"] = n

  # Sanitize graph_module to match in the same way as pattern's graph_module.
  graph_module_to_match = passes.remove_clone_ops(graph_module_to_match)

  match_with_attrs = pattern.match(graph_module_to_match)

  for match, attr in match_with_attrs:
    match_id = _get_uuid()

    # NOTE: Current graph rewriter (_insert_marker) does not work perfectly
    # with continuous matches e.g. matching (a + b) on (w + x + y + z). The
    # rewritten results may be undetermined with false negative - some
    # matches may not be marked in the lowering, while the marked ones would
    # always be correct.
    # TODO(cnchan): completely support mark_pattern on continuous matches.
    for i, pattern_input_node in enumerate(pattern.input_nodes):
      input_node = match.nodes_map[pattern_input_node]
      new_input_node = _insert_marker(
          graph_module,
          input_node.meta["ORIGINAL_NODE"],
          name=pattern.name,
          pos=i,
          id=match_id,
          is_input=True,
      )

      # Only replace input by the marker node for those nodes used in the pattern.
      in_pattern_nodes = set(match.nodes_map.values())
      for user in input_node.users.keys():
        if user in in_pattern_nodes:
          user.meta["ORIGINAL_NODE"].replace_input_with(
              input_node.meta["ORIGINAL_NODE"], new_input_node
          )

    for i, pattern_output_node in enumerate(pattern.output_nodes):
      output_node = match.nodes_map[pattern_output_node]
      new_output_node = _insert_marker(
          graph_module,
          output_node.meta["ORIGINAL_NODE"],
          name=pattern.name,
          pos=i,
          id=match_id,
          is_input=False,
          attr=attr,  # torch_xla internal: only output marker needs attr.
      )
      output_node.meta["ORIGINAL_NODE"].replace_all_uses_with(new_output_node)
      new_output_node.update_arg(0, output_node.meta["ORIGINAL_NODE"])

  graph_module.graph.eliminate_dead_code()
  _prepose_placeholder_nodes(graph_module.graph)

  graph_module.graph.lint()
  graph_module.recompile()
  return graph_module
