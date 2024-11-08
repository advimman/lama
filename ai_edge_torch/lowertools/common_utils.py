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

import logging
from typing import List

from ai_edge_torch._convert import signature as signature_module
import tensorflow as tf
import torch
import torch.utils._pytree as pytree


def _flatten_list(l: List) -> List:
  flattened = []
  for item in l:
    if isinstance(item, list):
      flattened.extend(_flatten_list(item))
    else:
      flattened.append(item)
  return flattened


def flat_dict_names(
    tree_spec: pytree.TreeSpec, context: pytree.Context
) -> List[str]:
  """Given a TreeSpec, this produces a list of names for the leaves.

  The list of names embeddeds the structure of the tree_spec. A nesting level is
  indicated by an `_` and elements in a list are indicated by `_<index>`.

  TODO b/361601485: The flattening of names is not collision-free and needs to
  be revised.

  Args:
    tree_spec: The TreeSpec to extract the names from.
    context: The context used to check if the provided spec belongs to a
      dictionary or a list.

  Returns:
    A list of flattened names.
  """
  flat_names = []
  if context is None:
    for i, spec in enumerate(tree_spec):
      if spec.children_specs:
        flat_names.extend([
            f"{i}_{name}"
            for name in flat_dict_names(spec.children_specs, spec.context)
        ])
      else:
        flat_names.append(f"{i}")
  else:
    flat_ctx = _flatten_list(context)
    for prefix, spec in zip(flat_ctx, tree_spec):
      leaf_flat_names = flat_dict_names(spec.children_specs, spec.context)
      if leaf_flat_names:
        flat_names.extend([f"{prefix}_{name}" for name in leaf_flat_names])
      else:
        flat_names.append(prefix)

  return flat_names


def _torch_to_tf_variable(torch_tensor: torch.Tensor):
  if not torch_tensor.is_contiguous():
    torch_tensor = torch_tensor.contiguous()

  try:
    dlpack_capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
    tf_tensor = tf.experimental.dlpack.from_dlpack(dlpack_capsule)
  except Exception:
    logging.info(
        "Can not use dlpack to convert torch tensors. Falling back to numpy."
    )
    nparray = torch_tensor.cpu().detach().numpy()
    tf_tensor = tf.convert_to_tensor(nparray)

  return tf.Variable(tf_tensor, trainable=False)


def _get_states(
    exported_programs: list[torch.export.ExportedProgram],
    signatures: list[signature_module.Signature],
):
  for exported_program, signature in zip(exported_programs, signatures):
    args, _ = exported_program.example_inputs
    # Calling this to get **all** the state including model buffers.
    _flat_input_args = exported_program._graph_module_flat_inputs(args, {})
    for tensor, input_spec in zip(
        _flat_input_args, exported_program.graph_signature.input_specs
    ):
      # Only interested in Tensors that are part of the state (and not user input).
      if (
          not isinstance(tensor, torch.Tensor)
          or input_spec.kind
          == torch.export.graph_signature.InputKind.USER_INPUT
      ):
        continue
      yield signature, tensor, input_spec


def _tensor_unique_id(tensor: torch.Tensor):
  return (
      str(tensor.device),
      tensor.shape,
      tensor.stride(),
      tensor.untyped_storage().data_ptr(),
  )


def gather_state_dict(
    exported_programs: list[torch.export.ExportedProgram],
    signatures: list[signature_module.Signature],
):
  deduped_tensor_map = {}

  for _, tensor, _ in _get_states(exported_programs, signatures):
    unique_id = _tensor_unique_id(tensor)
    deduped_tensor_map[unique_id] = _torch_to_tf_variable(tensor)

  state_dict = {}
  for signature, tensor, input_spec in _get_states(
      exported_programs, signatures
  ):
    unique_id = _tensor_unique_id(tensor)
    state_dict[signature.name + "_" + input_spec.target] = deduped_tensor_map[
        unique_id
    ]

  return state_dict, list(deduped_tensor_map.values())
