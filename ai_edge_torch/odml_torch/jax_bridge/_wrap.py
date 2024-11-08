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
"""APIs to wrap JAX functions for using in ODML Torch lowerings."""

import functools
import inspect
from typing import Any, Callable, cast
import uuid
from ai_edge_torch.odml_torch import export_utils
from ai_edge_torch.odml_torch import passes
from ai_edge_torch.odml_torch.jax_bridge import utils
import jax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func
import torch.utils._pytree as pytree

# Jax double (64bit) precision is required to generate StableHLO mlir with
# i64/f64 tensors from Jax bridged lowerings. If not set properly, all the
# 64bit tensors would be truncated to 32bit dtype and potentially break the
# lowering.
jax.config.update("jax_enable_x64", True)


def _lower_to_ir_text(
    jaxfn, args, kwargs, ir_input_names: list[str] = None
) -> tuple[str, list[ir.Value]]:
  args = utils.tree_map_list_to_tuple(args)
  kwargs = utils.tree_map_list_to_tuple(kwargs)

  names_args = [
      *zip(inspect.signature(jaxfn).parameters.keys(), args),
      *kwargs.items(),
  ]

  static_argnames = []
  jax_lower_static_kwargs = {}
  jax_lower_args = []
  jax_lower_argnames = []
  ir_inputs = []

  for i, (name, arg) in enumerate(names_args):
    is_positional = i < len(args)
    if not utils.is_ir_variable(arg):
      static_argnames.append(name)
      jax_lower_static_kwargs[name] = arg
    else:
      # Enforce the arg order in the mlir is the same as the lowering func
      jax_lower_args.append(utils.ir_variable_to_jax(arg))

      if is_positional and len(jax_lower_args) == i + 1:
        # The first N continuous tensor args are passed to the lowering func
        # as positional args, when they passed to the bridged func as
        # positional args also.
        jax_lower_argnames.append(None)
      else:
        # Otherwise pass the arg to the lowering func as keyword arg.
        jax_lower_argnames.append(name)

      if ir_input_names is None or name in ir_input_names:
        # ir variable can be a nested tuple, while mlir args should be flat.
        ir_inputs += [
            x for x in pytree.tree_flatten(arg)[0] if isinstance(x, ir.Value)
        ]

  def lower_wrapper(*args):
    nonlocal jax_lower_static_kwargs

    jaxfn_args = []
    jaxfn_kwargs = jax_lower_static_kwargs.copy()
    for name, arg in zip(jax_lower_argnames, args):
      if name is None:
        jaxfn_args.append(arg)
      else:
        jaxfn_kwargs[name] = arg

    return jaxfn(*jaxfn_args, **jaxfn_kwargs)

  return jax.jit(lower_wrapper).lower(*jax_lower_args).as_text(), ir_inputs


def wrap(jaxfn: Callable[Any, Any], ir_input_names: list[str] = None):
  """Return the wrapped JAX function to be used in ODMLTorch lowerings.

  If the given jaxfn has signature `jaxfn(*args, **kwargs) -> return`, the
  wrapped function would:
  - Have signature `wrapped(lctx: odml_torch.export.LoweringContext, *args,
  **kwargs) -> return`.
  - Accept mlir.ir.Value for all params expecting jax.Array as inputs.
  - Return mlir.ir.Value for all jax.Array outputs from jaxfn.

  Args:
    jaxfn: The JAX function to be wrapped.
    ir_input_names: The input (param) names of the JAX function to be used in
      the MLIR lowering. This is useful when the JAX impl only depends on
      specific inputs to the function. If not specified, all ir.Value passed to
      the wrapped function are assumed to be used in the lowering.
  """

  @functools.wraps(jaxfn)
  def wrapped(lctx, *args, **kwargs):

    ir_text, ir_inputs = _lower_to_ir_text(
        jaxfn,
        args,
        kwargs,
        ir_input_names=ir_input_names,
    )

    module = ir.Module.parse(ir_text)
    passes.strip_debuginfo(module)

    symbol_table = ir.SymbolTable(module.operation)
    main_func = symbol_table["main"]

    with ir.InsertionPoint.at_block_begin(lctx.ir_module.body):
      cloned_func = cast(func.FuncOp, main_func.clone())
      cloned_func_name = f"{jaxfn.__name__}_{uuid.uuid4().hex[:8]}"
      cloned_func.attributes["sym_name"] = ir.StringAttr.get(cloned_func_name)
      cloned_func.attributes["sym_visibility"] = ir.StringAttr.get("private")

    # HACK: Use the custom inliner implemented in Python because MLIR inline
    # pass from JAX's MLIR pybinding build in OSS cannot properly inline
    # func.call ops.
    # This should be switched to `passes.inline(module)` when we have our own
    # MLIR pybinding build.
    export_utils.inline(symbol_table, cloned_func.entry_block)

    if not cloned_func.arguments:
      # Known edge case: when the lowering does not depend on input but
      # just the meta of input like shape or dtype.
      ir_inputs = []

    results = func.CallOp(cloned_func, ir_inputs).results
    if len(results) == 1:
      return results[0]
    return results

  return wrapped
