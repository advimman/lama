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
"""Layout mark for the optimized layout transposes pass."""

import torch

# Tag which is added to a node's meta to indicate that is is part of the NHWC
# partition.
IS_NHWC_NODE = "OPTIMIZE_LAYOUT_TRANSPOSES_PASS__IS_NHWC_NODE"


# Tag which is added to a node's meta to indicate that it is derived completely
# from constant and/or weight tensor(s).
IS_CONST_NODE = "OPTIMIZE_LAYOUT_TRANSPOSES_PASS__IS_CONST_NODE"


def mark_as_nhwc_node(node: torch.fx.Node) -> None:
  node.meta[IS_NHWC_NODE] = True


def mark_as_nchw_node(node: torch.fx.Node) -> None:
  node.meta[IS_NHWC_NODE] = False


def is_nhwc_node(node: torch.fx.Node) -> bool:
  return node.meta.get(IS_NHWC_NODE, False)


def is_nchw_node(node: torch.fx.Node) -> bool:
  return not is_nhwc_node(node)


def mark_as_const_node(node: torch.fx.Node) -> None:
  node.meta[IS_CONST_NODE] = True


def is_const_node(node: torch.fx.Node) -> bool:
  return node.meta.get(IS_CONST_NODE, False)
