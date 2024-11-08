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
"""Wrappers for latest torch APIs/utilities to maintain backward compatibility with older torch releases."""

import torch
from torch.fx import _pytree as fx_pytree


def graph_module_flat_inputs(ep: torch.export.ExportedProgram, args, kwargs):
  """Transform args, kwargs of __call__ to args for graph_module.

  self.graph_module takes stuff from state dict as inputs.
  The invariant is for ep: ExportedProgram is
  ep(args, kwargs) ==
    ep.postprocess(ep.graph_module(ep.graph_module_flat_inputs(args, kwargs)))
  """
  if hasattr(ep, "_graph_module_flat_inputs"):
    return ep._graph_module_flat_inputs(args, kwargs)

  if args is None:
    args = tuple()
  if kwargs is None:
    kwargs = {}

  flat_args = args
  if (in_spec := ep.call_spec.in_spec) is not None:
    if (
        in_spec.type == tuple
        and len(in_spec.children_specs) == 2
        and in_spec.children_specs[0].type == tuple
        and in_spec.children_specs[1].type == dict
    ):
      # NOTE: this is the case where in_spec is for both args and kwargs
      flat_args = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
    else:
      flat_args = fx_pytree.tree_flatten_spec(args, in_spec)

  param_buffer_keys = ep.graph_signature.parameters + ep.graph_signature.buffers
  param_buffer_values = tuple(ep.state_dict[key] for key in param_buffer_keys)

  if hasattr(ep.graph_signature, "lifted_tensor_constants"):
    ordered_tensor_constants = tuple(
        ep.tensor_constants[name]
        for name in ep.graph_signature.lifted_tensor_constants
    )
  else:
    ordered_tensor_constants = tuple()

  return (*param_buffer_values, *flat_args, *ordered_tensor_constants)


# TODO(b/331481564): Replace this with CanonicalizePass + run_decomposition
def safe_run_decompositions(exported_program, decomp_table=None):
  for node in exported_program.graph.nodes:
    if node.target == torch.ops.aten.view.default:
      # Passes or torch.export may generate aten.view nodes not respecting the
      # tensor memory format. Changes all the aten.view to torch.reshape
      # for retracing. If the input memory format is already contiguous,
      # retracing in run_decomposition below would decompose torch.reshape
      # back to one aten.view.
      node.target = lambda self, size: torch.reshape(self.contiguous(), size)

  return exported_program.run_decompositions(decomp_table)
