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
import torch


def _class_fullname(cls):
  module = cls.__module__
  if module == "builtins":
    return cls.__qualname__
  return module + "." + cls.__qualname__


def _get_hierarchy(node: torch.fx.Node):
  nn_module_stack = node.meta.get("nn_module_stack", {})
  layers = []
  for name, layer in nn_module_stack.values():
    iid = ("_" + name.split(".")[-1]) if name else ""
    layer_str = layer if isinstance(layer, str) else _class_fullname(layer)
    layers.append(layer_str + iid)

  hierachy_str = "/".join(layers) + ";"
  return hierachy_str


def build_mlir_debuginfo(node: torch.fx.Node):
  """Build the debuginfo string for the given node's lowerings in MLIR."""

  if not hasattr(node, "meta") or "nn_module_stack" not in node.meta:
    return None

  return _get_hierarchy(node)
