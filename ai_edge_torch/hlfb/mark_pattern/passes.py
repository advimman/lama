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
"""Passes to clean up the model graph for pattern matching."""

import torch


def remove_clone_ops(gm: torch.fx.GraphModule):
  """Removes clone ops from the graph.

  torch export adds additional aten.clone nodes to produce contiguous in memory
  tensors depending on tensor sizes for runtime efficiency. However, these
  unpredictable clone nodes can break the pattern matching. Thus remove all
  clones in model and pattern graphs.

  Args:
    gm: The graph module to remove clone ops from.

  Returns:
    The graph module with clone ops removed.
  """
  for node in gm.graph.nodes:
    if node.op == "call_function" and node.name.startswith("clone"):
      node.replace_all_uses_with(node.args[0])
      gm.graph.erase_node(node)

  gm.graph.lint()
  gm.recompile()
  return gm


def remove_dangling_args(gm: torch.fx.GraphModule):
  """Removes dangling args from the graph.

  Args:
    gm: The graph module to remove dangling args from.

  Returns:
    The graph module with dangling args removed.
  """
  nodes_to_erase = []
  for node in gm.graph.nodes:
    if node.op == "placeholder" and len(node.users) == 0:
      nodes_to_erase.append(node)
  for node in nodes_to_erase:
    gm.graph.erase_node(node)

  gm.graph.lint()
  gm.recompile()
  return gm
