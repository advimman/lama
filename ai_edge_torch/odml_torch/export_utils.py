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
"""Utilities for ODML Torch export."""

import functools
import re
from typing import Sequence, cast
import jax._src.interpreters.mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func
import torch

# std::numeric_limits<int64_t>::min()
IR_DYNAMIC = -9223372036854775808


def is_ir_dynamic(v):
  return v == IR_DYNAMIC


def is_torch_dynamic(v):
  return isinstance(v, torch.SymInt)


def is_iterable(v):
  try:
    iter(v)
  except TypeError:
    return False
  return True


def create_ir_context():
  # HACK: Use ir context from JAX as base for better stability in OSS.
  # TODO(b/362798610) Build MLIR pybinding in ai-edge-torch release.
  context = jax._src.interpreters.mlir.make_ir_context()
  context.allow_unregistered_dialects = True

  return context


def inline(
    symbol_table: ir.SymbolTable,
    block: ir.Block,
):
  """Recursively inlines all func.call ops in the block.

  The symbol_table must include all func.func called by func.call ops.
  This inliner in Python is implemented because MLIR inline pass from JAX's
  MLIR pybinding build in OSS cannot properly inline func.call ops.
  """
  while True:
    is_changed = False
    for op in block.operations:
      if op.OPERATION_NAME != func.CallOp.OPERATION_NAME:
        continue

      call_op = cast(func.CallOp, op)
      func_op = cast(func.FuncOp, symbol_table[call_op.callee.value])
      with ir.InsertionPoint(op):
        new_results = clone_func_body_ops(func_op, call_op.operands)

      for old_result, new_result in zip(call_op.results, new_results):
        old_result = cast(ir.Value, old_result)
        old_result.replace_all_uses_with(new_result)
      call_op.erase()
      is_changed = True

    if not is_changed:
      break

  for op in block.operations:
    for region in op.regions:
      for block in region.blocks:
        inline(symbol_table, block)


def clone_func_body_ops(func_op: func.FuncOp, ir_inputs: Sequence[ir.Value]):
  """Clone operations in the func_op's body by one into the current context."""
  func_args = list(func_op.arguments)
  ir_inputs = list(ir_inputs)
  assert len(func_args) == len(ir_inputs)

  value_mapping = {arg: ir_input for arg, ir_input in zip(func_args, ir_inputs)}

  for op in list(func_op.entry_block.operations):
    cloned_operands = [value_mapping[val] for val in op.operands]
    if op.OPERATION_NAME == func.ReturnOp.OPERATION_NAME:
      return cloned_operands

    cloned = cast(ir.Operation, op.operation.clone())

    for i in range(len(op.operands)):
      cloned.operands[i] = cloned_operands[i]

    for i in range(len(op.results)):
      value_mapping[op.results[i]] = cloned.results[i]

  return []


def sanitize_aten_op_name(op, chars=":."):
  return re.sub("[{}]".format(chars), "_", str(op))


def build_ir_attr(val):
  if val is None:
    return ir.StringAttr.get("py_None")
  if isinstance(val, bool):
    return ir.BoolAttr.get(val)
  if isinstance(val, int):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), val)
  if isinstance(val, float):
    return ir.BoolAttr.get(val)
  if isinstance(val, str):
    return ir.StringAttr.get(val)
  if isinstance(val, dict):
    return ir.DictAttr.get({k: build_ir_attr(v) for k, v in val.items()})
  if isinstance(val, (list, tuple)):
    return ir.ArrayAttr.get([build_ir_attr(v) for v in val])

  # Stringify the value to a StringAttr by default
  return ir.StringAttr.get(str(val))


def torch_dtype_to_ir_element_type(ctx, dtype):
  ty_get = {
      torch.double: ir.F64Type.get,
      torch.float32: ir.F32Type.get,
      torch.half: ir.F16Type.get,
      torch.long: functools.partial(ir.IntegerType.get_signless, 64),
      torch.int32: functools.partial(ir.IntegerType.get_signless, 32),
      torch.int16: functools.partial(ir.IntegerType.get_signless, 16),
      torch.bool: functools.partial(ir.IntegerType.get_signless, 1),
  }.get(dtype)
  return ty_get(ctx)


def ir_element_type_to_torch_dtype(ty):
  if isinstance(ty, ir.F32Type):
    return torch.float32
  if isinstance(ty, ir.F64Type):
    return torch.float64
  if isinstance(ty, ir.F16Type):
    return torch.half
  if isinstance(ty, ir.IntegerType):
    if ty.is_signless:
      if ty.width == 64:
        return torch.long
      if ty.width == 32:
        return torch.int32
      if ty.width == 16:
        return torch.int16
      if ty.width == 1:
        return torch.bool
  raise RuntimeError(f"Unsupported ir element type: {ty}")
