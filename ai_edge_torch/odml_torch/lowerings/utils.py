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
"""Utilities for building MLIR lowerings."""

import numbers
from typing import Any
from typing import Optional

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import numpy as np


def splat(val, ty, shape=tuple(), *, loc: Optional[Any] = None):
  if isinstance(ty, ir.IntegerType):
    if ty.width == 1:
      attr = ir.BoolAttr.get(bool(val))
    else:
      attr = ir.IntegerAttr.get(ty, int(val))
  elif isinstance(ty, ir.FloatType):
    attr = ir.FloatAttr.get(ty, val)
  else:
    raise ValueError("Unsupported type: %s" % str(ty))

  return stablehlo.constant(
      ir.DenseElementsAttr.get_splat(
          ir.RankedTensorType.get(shape, ty),
          attr,
      ),
      loc=loc,
  )


def get_common_broadcast_shape(
    shape_1: list[int], shape_2: list[int]
) -> Optional[list[int]]:
  if not shape_1 and not shape_2:
    return None

  shape_1 = shape_1 if shape_1 else [1]
  shape_2 = shape_2 if shape_2 else [1]

  length_diff = abs(len(shape_1) - len(shape_2))
  if len(shape_1) < len(shape_2):
    shape_1 = [1] * length_diff + shape_1
  elif len(shape_1) > len(shape_2):
    shape_2 = [1] * length_diff + shape_2

  common_broadcast_shape = []
  for idx in reversed(range(len(shape_1))):
    dim_size1 = shape_1[idx]
    dim_size2 = shape_2[idx]

    if dim_size1 == dim_size2:
      common_broadcast_shape.insert(0, dim_size1)
    elif dim_size1 == 1 or dim_size2 == 1:
      common_broadcast_shape.insert(0, max(dim_size1, dim_size2))
    else:
      return None

  return common_broadcast_shape


def get_broadcast_dimensions(
    shape_from: list[int], shape_to: list[int]
) -> list[int]:
  assert get_common_broadcast_shape(shape_from, shape_to) == shape_to

  ret = []
  for val in range(len(shape_to) - len(shape_from), len(shape_to)):
    ret.append(val)

  return ir.DenseI64ArrayAttr.get(np.asarray(ret, np.int64))


def broadcast_args_if_needed(
    val_1: ir.Value, val_2: ir.Value
) -> tuple[Optional[ir.Value], Optional[ir.Value]]:
  broadcast_shape = get_common_broadcast_shape(
      val_1.type.shape, val_2.type.shape
  )
  if broadcast_shape is None:
    return None, None

  new_val_1, new_val_2 = val_1, val_2

  if val_1.type.shape != broadcast_shape:
    new_val_1 = stablehlo.broadcast_in_dim(
        result=ir.RankedTensorType.get(
            broadcast_shape, val_1.type.element_type
        ),
        operand=val_1,
        broadcast_dimensions=get_broadcast_dimensions(
            val_1.type.shape, broadcast_shape
        ),
    )
  if val_2.type.shape != broadcast_shape:
    new_val_2 = stablehlo.broadcast_in_dim(
        result=ir.RankedTensorType.get(
            broadcast_shape, val_2.type.element_type
        ),
        operand=val_2,
        broadcast_dimensions=get_broadcast_dimensions(
            val_2.type.shape, broadcast_shape
        ),
    )
  return new_val_1, new_val_2


def upcast_to_same_type(*vals: ir.Value):
  if not vals:
    return None
  if len(vals) == 1:
    return vals[0]

  def get_priority(ty: ir.Type):
    priorities = [
        ir.IntegerType.get_signless(1),
        ir.IntegerType.get_signless(16),
        ir.IntegerType.get_signless(32),
        ir.IntegerType.get_signless(64),
        ir.F16Type,
        ir.F32Type,
        ir.F64Type,
    ]
    for i, tycls in enumerate(priorities):
      if tycls.isinstance(ty):
        return i
    raise ValueError("Unsupported type: %s" % str(ty))

  cast_tycls = type(max([v.type.element_type for v in vals], key=get_priority))
  new_vals = []
  for val in vals:
    if not cast_tycls.isinstance(val.type.element_type):
      val = stablehlo.convert(
          ir.RankedTensorType.get(val.type.shape, cast_tycls.get()), val
      )
    new_vals.append(val)
  return tuple(new_vals)


def minmax(ty: ir.Type) -> tuple[numbers.Number, numbers.Number]:
  if isinstance(ty, ir.IntegerType):
    if ty.is_unsigned:
      return (0, 1 << ty.width)
    else:
      return (-(1 << (ty.width - 1)), (1 << (ty.width - 1)) - 1)
  elif isinstance(ty, ir.F16Type):
    return (np.finfo(np.float16).min, np.finfo(np.float16).max)
  elif isinstance(ty, ir.F32Type):
    return (np.finfo(np.float32).min, np.finfo(np.float32).max)
  elif isinstance(ty, ir.F64Type):
    return (np.finfo(np.float64).min, np.finfo(np.float64).max)
  else:
    raise ValueError("Unsupported type: %s" % ty)


def convert_int_to_float(t: ir.Value) -> ir.Value:
  """Converts an input with type ir.IntegerType to an ir.FloatType of equivalent width."""
  elty = t.type.element_type
  if not isinstance(elty, ir.IntegerType):
    raise ValueError(
        "Expected input with integer type, received %s" % type(elty)
    )

  if elty.width == 32:
    return stablehlo.convert(
        ir.RankedTensorType.get(t.type.shape, ir.F32Type.get()), t
    )
  elif elty.width == 64:
    return stablehlo.convert(
        ir.RankedTensorType.get(t.type.shape, ir.F64Type.get()), t
    )
