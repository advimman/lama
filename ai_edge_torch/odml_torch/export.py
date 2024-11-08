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
"""APIs to convert and lower a PyTorch ExportedProgram to MLIR."""

import dataclasses
import enum
import io
import operator
from typing import Any, Callable, Optional

from jax.lib import xla_extension
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch
import torch.utils._pytree as pytree

from . import _torch_future
from . import debuginfo
from . import export_utils
from . import lowerings

LoweringContext = lowerings.context.LoweringContext


def _build_flat_inputs(
    ctx: ir.Context, exported_program: torch.export.ExportedProgram
):
  """Build flattened inputs and metadata from exported program's signature."""
  placeholder_nodes = [
      n for n in exported_program.graph.nodes if n.op == "placeholder"
  ]
  export_flat_args = _torch_future.graph_module_flat_inputs(
      exported_program, *exported_program.example_inputs
  )

  ir_inputs = []
  tensor_metas = []
  for node, arg in zip(placeholder_nodes, export_flat_args):
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
      raise RuntimeError(f"{type(arg)} (for {node.name}) is not a tensor")

    tensor_metas.append(tensor_meta)
    # Assume all dynamic dimensions are unbounded.
    # TODO: Add checks for ep.range_constraints in MLIR.
    shape = tuple(
        export_utils.IR_DYNAMIC if export_utils.is_torch_dynamic(s) else s
        for s in tensor_meta.shape
    )
    ir_inputs.append(
        ir.RankedTensorType.get(
            shape,
            export_utils.torch_dtype_to_ir_element_type(ctx, tensor_meta.dtype),
        )
    )
  return tuple(ir_inputs), tuple(export_flat_args), tuple(tensor_metas)


def _get_output_metas(exported_program: torch.export.ExportedProgram):
  """Get the output node's tensor_meta from the exported program."""
  outputs = [n for n in exported_program.graph.nodes if n.op == "output"]
  assert len(outputs) == 1
  outputs, _ = pytree.tree_flatten(outputs[0].args[0])
  assert all(isinstance(output, torch.fx.Node) for output in outputs)
  return tuple(output.meta["tensor_meta"] for output in outputs)


class LoweringInterpreter(torch.fx.Interpreter):
  """The FX interpreter to iterate and invoke corresponding lowering for each PyTorch op in the graph."""

  def __init__(self, module: torch.fx.GraphModule, lctx: LoweringContext):
    super().__init__(module)
    self.lctx = lctx
    self.outputs = None

  def _build_loc(self, node: torch.fx.Node):

    info = debuginfo.build_mlir_debuginfo(node)
    if info is None:
      return ir.Location.unknown()

    return ir.Location.name(name=info)

  def run_node(self, node: torch.fx.Node):
    loc = self._build_loc(node)
    with loc:
      self.lctx = self.lctx.replace(ir_location=loc, node=node)
      res = super().run_node(node)
      self.lctx = self.lctx.replace(ir_location=None, node=None)
    return res

  def call_function(self, target, args, kwargs):
    if target is operator.getitem:
      return super().call_function(target, args, kwargs)

    if hasattr(target, "_schema"):
      new_args = []
      for arg, spec in zip(args, target._schema.arguments):
        if isinstance(spec.type, torch.TensorType):
          if isinstance(arg, int):
            arg = lowerings.utils.splat(arg, ir.IntegerType.get_signless(32))
          elif isinstance(arg, float):
            arg = lowerings.utils.splat(arg, ir.F32Type.get())

        new_args.append(arg)
      args = tuple(new_args)

    lowering = lowerings.lookup(target)
    if lowering is None:
      raise RuntimeError(f"Lowering not found: {target}")
    return lowering(self.lctx, *args, **kwargs)

  def output(self, target, args, kwargs):
    flat_outputs = pytree.tree_flatten(args[0])[0]
    self.outputs = flat_outputs


@dataclasses.dataclass
class InputSpec:

  class VariableType(enum.Enum):
    USER_INPUT = "user_input"
    PARAMETER = "parameter"

  type_: VariableType
  i: int = -1
  name: str = ""

  @classmethod
  def parameter(cls, name: str):
    return cls(type_=cls.VariableType.PARAMETER, name=name)

  @classmethod
  def user_input(cls, i: int):
    return cls(type_=cls.VariableType.USER_INPUT, i=i)

  @property
  def is_parameter(self):
    return self.type_ == self.VariableType.PARAMETER

  @property
  def is_user_input(self):
    return self.type_ == self.VariableType.USER_INPUT


@dataclasses.dataclass
class VariableSignature:  # either argument or parameters
  shape: list[int]
  dtype: str
  input_spec: InputSpec = None


@dataclasses.dataclass
class MlirLowered:
  """The lowered MLIR module, metadata, and weight tensors bundle from exported program."""

  ctx: ir.Context
  module: ir.Module
  state_dict: dict[str, torch.Tensor]
  input_signature: list[VariableSignature]
  output_signature: list[VariableSignature]

  _tf_function: Optional[Callable[Any, Any]] = None

  def __str__(self):
    return str(self.get_text(enable_debug_info=False))

  def __repr__(self):
    return str(self.get_text(enable_debug_info=False))

  def get_text(self, enable_debug_info=False):
    return str(
        self.module.operation.get_asm(enable_debug_info=enable_debug_info)
    )

  @property
  def module_bytecode(self) -> bytes:
    output = io.BytesIO()
    self.module.operation.write_bytecode(file=output)
    return output.getvalue()

  @property
  def module_bytecode_vhlo(self) -> bytes:
    # HACK: In OSS, we use MLIR pybinding and StableHLO dialect from JAX's
    # build, which may not have the same StableHLO version as what used in
    # TFLite converter. Therefore we always serialize MLIR module in VHLO.
    # TODO(b/362798610) Build MLIR pybinding in ai-edge-torch release.
    target_version = stablehlo.get_minimum_version()
    module_bytecode = xla_extension.mlir.serialize_portable_artifact(
        self.module_bytecode, target_version
    )
    return module_bytecode

  @property
  def tf_function(self):
    # Lazy import
    from . import tf_integration

    if self._tf_function is None:
      self._tf_function = tf_integration.mlir_to_tf_function(self)
    return self._tf_function

  def __call__(self, *args):
    # Lazy importing TF when execution is needed.
    return self.tf_function(*args)

  def to_flatbuffer(self):
    from . import tf_integration

    return tf_integration.mlir_to_flatbuffer(self)


# TODO(b/331481564) Make this a ai_edge_torch FX pass.
def _convert_i64_to_i32(exported_program: torch.export.ExportedProgram):
  """Convert internal constant aten ops' output from int64 to int32.

  Int32 generally has better performance and compatibility than int64 in
  runtime. This pass converts aten op where the output(s) are int64 constant
  tensors to return int32 constant tensors.

  Args:
    exported_program: The exported program to apply the pass.
  """

  def in_i32(x: int):
    return -2147483648 <= x <= 2147483647

  def rewrite_arange(node: torch.fx.Node):
    tensor_meta = node.meta.get("tensor_meta", None)
    if not tensor_meta:
      return

    start, end = node.args[:2]
    if tensor_meta.dtype != torch.int64:
      return
    if not (in_i32(start) and in_i32(end)):
      return
    op = node.target
    node.target = lambda *args, **kwargs: op(*args, **kwargs).type(torch.int32)

  graph_module = exported_program.graph_module
  for node in graph_module.graph.nodes:

    if node.target == torch.ops.aten.arange.start_step:
      rewrite_arange(node)


def exported_program_to_mlir(
    exported_program: torch.export.ExportedProgram,
) -> MlirLowered:
  """Lower the exported program to MLIR."""
  exported_program = _torch_future.safe_run_decompositions(
      exported_program, lowerings.decompositions()
  )

  _convert_i64_to_i32(exported_program)
  exported_program = _torch_future.safe_run_decompositions(
      exported_program, lowerings.decompositions()
  )

  with export_utils.create_ir_context() as context, ir.Location.unknown():

    module = ir.Module.create()
    lctx = LoweringContext(context, module)
    interpreter = LoweringInterpreter(exported_program.graph_module, lctx)
    ir_flat_inputs, export_flat_args, tensor_metas = _build_flat_inputs(
        context, exported_program
    )

    # HACK: OSS MLIR pybinding could mysteriously transform func.func under
    # construction into a func.return op after calling ir.Module.parse(..)
    # in the context, which happens in JAX bridge. This is a bug in MLIR
    # pybinding.
    # Workaround steps:
    # 1. Create a temp func.func.
    # 2. Create and insert ops to temp's entry block. During the process
    #   the temp func.func would be broken, but the ops in the block are fine.
    # 3. Create the main func.func and copy all the ops in temp's entry block
    #   to main.
    # 4. Erase the temp func.func.
    temp_func = func.FuncOp(
        "temp",
        ir.FunctionType.get(ir_flat_inputs, []),
        ip=ir.InsertionPoint.at_block_begin(module.body),
    )
    with ir.InsertionPoint(temp_func.add_entry_block()):
      interpreter.run(*temp_func.arguments, enable_io_processing=False)
      num_mutations = len(exported_program.graph_signature.buffers_to_mutate)
      outputs = interpreter.outputs[num_mutations:]
      func.ReturnOp(interpreter.outputs[num_mutations:])

    main_func = func.FuncOp(
        "main",
        ir.FunctionType.get(ir_flat_inputs, [o.type for o in outputs]),
        ip=ir.InsertionPoint.at_block_begin(module.body),
    )
    with ir.InsertionPoint(main_func.add_entry_block()):
      outputs = export_utils.clone_func_body_ops(temp_func, main_func.arguments)
      func.ReturnOp(outputs)

    main_func.attributes["sym_visibility"] = ir.StringAttr.get("public")
    temp_func.erase()

    module.operation.verify()

  input_signature = []
  state_dict = {}

  user_inputs_cnt = 0
  for arg, tensor_meta, input_spec in zip(
      export_flat_args,
      tensor_metas,
      exported_program.graph_signature.input_specs,
  ):
    # Assumption:
    # All states comes first in the list of args, and user provided inputs
    # comes later. Also there is no kwargs.
    if input_spec.kind == torch.export.graph_signature.InputKind.USER_INPUT:
      input_signature.append(
          VariableSignature(
              tensor_meta.shape,
              tensor_meta.dtype,
              input_spec=InputSpec.user_input(user_inputs_cnt),
          )
      )
      user_inputs_cnt += 1
    else:
      # Parameter or constant
      state_dict[input_spec.target] = arg
      input_signature.append(
          VariableSignature(
              tensor_meta.shape,
              tensor_meta.dtype,
              input_spec=InputSpec.parameter(input_spec.target),
          )
      )

  output_signature = [
      VariableSignature(tensor_meta.shape, tensor_meta.dtype)
      for tensor_meta in _get_output_metas(exported_program)
  ]
  return MlirLowered(
      context, module, state_dict, input_signature, output_signature
  )
