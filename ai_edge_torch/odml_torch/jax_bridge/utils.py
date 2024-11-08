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
"""Utilities for Jax bridge."""

from ai_edge_torch import odml_torch
import jax
import jax.numpy as jnp
from jax._src.lib.mlir import ir
import torch


def t2j_dtype(dtype):
  return {
      torch.bfloat16: jnp.bfloat16,
      torch.half: jnp.float16,
      torch.float32: jnp.float32,
      torch.double: jnp.double,
      torch.long: jnp.int64,
      torch.int64: jnp.int64,
      torch.int32: jnp.int32,
      torch.int16: jnp.int16,
      torch.int8: jnp.int8,
      torch.uint8: jnp.uint8,
      torch.bool: jnp.bool_,
      torch.complex64: jnp.complex64,
      torch.complex128: jnp.complex128,
  }.get(dtype)


def is_ir_variable(value):
  if isinstance(value, ir.Value):
    return True
  if isinstance(value, (list, tuple)):
    return any(is_ir_variable(x) for x in value)
  return False


def ir_variable_to_jax(value):
  if isinstance(value, (list, tuple)):
    return tuple([ir_variable_to_jax(x) for x in value])
  elif not isinstance(value, ir.Value):
    return value
  elif not isinstance(value.type, ir.RankedTensorType):
    raise ValueError(
        f"ir.Value to JAX must be in ir.RankedTensorType, got {value}"
    )

  return jax.ShapeDtypeStruct(
      value.type.shape,
      t2j_dtype(
          odml_torch.export_utils.ir_element_type_to_torch_dtype(
              value.type.element_type
          )
      ),
  )


def tree_map_list_to_tuple(value):
  if isinstance(value, dict):
    return {k: tree_map_list_to_tuple(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return tuple([tree_map_list_to_tuple(v) for v in value])
  return value
