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

from ai_edge_torch import fx_pass_base
from ai_edge_torch import lowertools
import torch
import torch.utils._pytree as pytree


def _get_mlir_debuginfo(node: torch.fx.Node):
  def class_fullname(cls):
    module = cls.__module__
    if module == "builtins":
      return cls.__qualname__
    return module + "." + cls.__qualname__

  def get_hierarchy(node: torch.fx.Node):
    nn_module_stack = node.meta.get("nn_module_stack", {})
    layers = []
    for name, layer in nn_module_stack.values():
      iid = ("_" + name.split(".")[-1]) if name else ""
      layer_str = layer if isinstance(layer, str) else class_fullname(layer)
      layers.append(layer_str + iid)

    hierachy_str = "/".join(layers) + ";"
    return hierachy_str

  # TODO(yijieyang): Encode aten op and attrs.
  return get_hierarchy(node)


def _wrap_call_function_node_with_debuginfo_writer(node: torch.fx.GraphModule):
  if not node.op.startswith("call_function"):
    return

  target = node.target
  debuginfo = _get_mlir_debuginfo(node)

  def debuginfo_writer(*args, **kwargs):
    nonlocal target, debuginfo
    outputs = target(*args, **kwargs)
    outputs = pytree.tree_map_only(
        torch.Tensor,
        lambda x: lowertools.write_mlir_debuginfo_op(x, debuginfo),
        outputs,
    )
    return outputs

  node.target = debuginfo_writer


class InjectMlirDebuginfoPass(fx_pass_base.PassBase):

  def call(self, graph_module: torch.fx.GraphModule):
    for node in graph_module.graph.nodes:
      _wrap_call_function_node_with_debuginfo_writer(node)

    graph_module.graph.lint()
    graph_module.recompile()
    return fx_pass_base.PassResult(graph_module, True)
