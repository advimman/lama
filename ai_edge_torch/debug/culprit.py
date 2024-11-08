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
"""Culprit finder for AI Edge Torch conversion."""

import contextlib
import copy
import dataclasses
import functools
import io
import operator
import os
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import ai_edge_torch
from ai_edge_torch.debug import utils
import torch
from torch._functorch import aot_autograd
from torch._functorch.fx_minifier import minifier as fx_minifier
import torch.utils._pytree as pytree

_torch_float_dtypes = {
    torch.float32,
    torch.float,
    torch.float64,
    torch.double,
    torch.float16,
    torch.half,
    torch.bfloat16,
}
_torch_int_dtypes = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.short,
    torch.int32,
    torch.int,
    torch.int64,
    torch.long,
}

_fx_op_runner = {
    "call_function": lambda target, args, kwargs: target(*args, **kwargs),
    "call_method": lambda target, args, kwargs: getattr(args[0], target)(
        *args[1:], **kwargs
    ),
}

_CULPRIT_GRAPH_MODULE_NAME = "CulpritGraphModule"


def _get_shape_str(t: torch.Tensor):
  return f"({', '.join(map(str, t.shape))},)"


def _tensor_to_random_tensor_call(t: torch.Tensor):
  shape_str = _get_shape_str(t)
  if t.dtype in _torch_float_dtypes:
    return f"torch.randn({shape_str}, dtype={t.dtype})"
  elif t.dtype in _torch_int_dtypes:
    return f"torch.randint(0, 10, {shape_str}, dtype={t.dtype})"
  elif t.dtype == torch.bool:
    return f"torch.randint(0, 2, {shape_str}, dtype={t.dtype})"
  else:
    raise ValueError(f"Unsupported dtype: {t.dtype}")


def _tensor_to_buffer(t: torch.Tensor):
  buff = io.BytesIO()
  torch.save(t, buff)
  buff.seek(0)
  return buff.read()


@dataclasses.dataclass
class SearchResult:
  graph_module: torch.fx.GraphModule
  inputs: Tuple[Any]

  @property
  def graph(self) -> torch.fx.Graph:
    return self.graph_module.graph

  @graph.setter
  def graph(self, fx_g: torch.fx.Graph):
    self.graph_module.graph = fx_g


@dataclasses.dataclass
class Culprit(SearchResult):
  _runtime_errors: bool

  @property
  def stack_traces(self) -> List[str]:
    stack_traces = set()
    for node in self.graph.nodes:
      if node.op.startswith("call_") and "stack_trace" in node.meta:
        stack_traces.add(node.meta["stack_trace"])
    return list(stack_traces)

  def print_readable(self, print_output=True):
    """Print the Python code for culprit graph module and sample args.

    Args:
      print_output: bool - If true, prints the code to stdout. Otherwise returns
        the code in a str.
    """
    # TODO: b/321263453 - Support Python code gen with sample arg tensor values.
    random_inputs = True

    graph_module_code = self.graph_module.print_readable(
        print_output=False
    ).rstrip()

    input_strs = []
    for value in self.inputs:
      if torch.is_tensor(value):
        if not random_inputs:
          input_strs.append(
              f"# size={_get_shape_str(value)}, dtype={value.dtype}"
          )
          input_strs.append(
              f"torch.load(io.BytesIO({_tensor_to_buffer(value)})),"
          )
        else:
          input_strs.append(_tensor_to_random_tensor_call(value) + ",")
      else:
        input_strs.append(str(value) + ",")

    inputs_code = (
        "_args = (\n"
        + "\n".join([" " * 4 + code for code in input_strs])
        + "\n)"
    )

    code = graph_module_code + "\n\n" + inputs_code
    if print_output:
      print(code)
    else:
      return code

  def print_code(self, print_output=True):
    """Print the Python code for culprit graph module, sample args, and AI

    Edge Torch conversion that will fail with the error.

    Args:
      print_output: bool - If true, prints the code to stdout. Otherwise returns
        the code in a str.
    """
    definitions = self.print_readable(print_output=False)
    code = (
        "import torch\n"
        + "from torch import device\n"
        + "import ai_edge_torch\n\n"
        + definitions
        + "\n\n_edge_model ="
        f" ai_edge_torch.convert({_CULPRIT_GRAPH_MODULE_NAME}().eval(),"
        " _args)\n"
    )
    if self._runtime_errors:
      code += "_edge_model(*_args)\n"

    if print_output:
      print(code)
    else:
      return code

  @property
  def code(self):
    return self.print_code(print_output=False)

  def __repr__(self):
    return self.print_readable(print_output=False)

  def __str__(self):
    return self.print_readable(print_output=False)


def _normalize_getitem_nodes(fx_gm: torch.fx.GraphModule):
  """This function turns all operator getitem nodes in ExportedProgram FX graph to

  new nodes composed of "computation + getitem". The normalization duplicates
  some computations in the graph but would make the graph more friendly for
  partitioning in FX minifier.
  """

  fx_gm = copy.deepcopy(fx_gm)
  graph = fx_gm.graph
  for n in graph.nodes:
    if n.target != operator.getitem:
      continue

    src_n, key = n.args
    if src_n.op not in _fx_op_runner:
      continue

    runner = _fx_op_runner.get(src_n.op)

    with graph.inserting_after(n):
      new_n = graph.call_function(
          lambda src_target, key, args, kwargs: operator.getitem(
              runner(src_target, args, kwargs), key
          ),
          (src_n.target, key, src_n.args, src_n.kwargs),
      )
      n.replace_all_uses_with(new_n)

  graph.eliminate_dead_code()
  fx_gm.graph = graph
  return fx_gm


def _erase_unused_inputs(
    fx_gm: torch.fx.GraphModule, inputs: Tuple[torch.Tensor]
):
  fx_gm = copy.deepcopy(fx_gm)
  inputs = tuple(inputs)
  args = fx_gm.graph.process_inputs(*inputs)
  args_iter = iter(args)

  graph = fx_gm.graph
  new_inputs = []
  for n in graph.nodes:
    if n.op == "placeholder":
      if n.target.startswith("*"):
        new_inputs += list(args_iter)
      elif len(n.users) > 0:
        new_inputs.append(next(args_iter))
      else:
        graph.erase_node(n)
        next(args_iter)
  new_inputs = tuple(new_inputs)
  fx_gm.graph = graph
  return fx_gm, new_inputs


def _lift_dead_ops_to_outputs(fx_gm: torch.fx.GraphModule):
  fx_gm = copy.deepcopy(fx_gm)

  new_outputs = []
  graph = fx_gm.graph
  nodes = list(graph.nodes)
  assert nodes[-1].op == "output" and sum(n.op == "output" for n in nodes) == 1
  for node in nodes:
    if node.op not in ("placeholder", "output") and len(node.users) == 0:
      new_outputs.append(node)

  output_node = nodes[-1]
  # FX output node returns the first arg as is.
  # ref: https://github.com/pytorch/pytorch/blob/1a578df57cc0f417f671634e564c62ef5d9a97e2/torch/fx/interpreter.py#L337
  new_outputs, _ = pytree.tree_flatten([new_outputs, output_node.args[0]])
  output_node.update_arg(0, tuple(new_outputs))

  fx_gm.graph = graph
  return fx_gm


def _erase_trivial_outputs(fx_gm: torch.fx.GraphModule):
  """Remove output nodes directly connected to an input node."""
  fx_gm = copy.deepcopy(fx_gm)

  graph = fx_gm.graph
  nodes = list(graph.nodes)
  assert nodes[-1].op == "output" and sum(n.op == "output" for n in nodes) == 1
  output_node = nodes[-1]

  outputs, _ = pytree.tree_flatten(output_node.args[0])
  new_outputs = [output for output in outputs if output.op != "placeholder"]
  output_node.update_arg(0, tuple(new_outputs))

  fx_gm.recompile()
  return fx_gm


def _erase_sub_gm_from_gm(
    fx_gm: torch.fx.GraphModule,
    fx_inputs: Tuple[torch.Tensor],
    sub_gm: torch.fx.GraphModule,
    sub_inputs: Tuple[torch.Tensor],
):
  fx_gm = copy.deepcopy(fx_gm)
  fx_inputs = list(fx_inputs)

  class EraseNodeInterpreter(torch.fx.Interpreter):

    def run_node(self, node):
      nonlocal fx_gm, fx_inputs
      res = super().run_node(node)
      if node.op not in ("placeholder", "output"):
        to_erase = next(m for m in fx_gm.graph.nodes if m.name == node.name)
        # Raise the output (tensor) of the erased node to be an input of
        # the new model graph. Some raised inputs may become unused later
        # when all the users are within the erased subgraph, those inputs
        # will be removed by the followed `_erase_unused_inputs` pass.
        with fx_gm.graph.inserting_before(to_erase):
          new_input = fx_gm.graph.placeholder(node.name + "__value")
        to_erase.replace_all_uses_with(new_input)

        fx_gm.graph.erase_node(to_erase)
        fx_inputs.append(res)
      return res

  interpreter = EraseNodeInterpreter(sub_gm)
  interpreter.run(*sub_inputs)

  fx_gm.graph.lint()
  fx_gm.recompile()

  # Ops prior to the erased subgraph may be dangling. Lift them as outputs.
  fx_gm = _lift_dead_ops_to_outputs(fx_gm)
  fx_gm = _erase_trivial_outputs(fx_gm)
  fx_gm, fx_inputs = _erase_unused_inputs(fx_gm, fx_inputs)

  fx_gm.graph.lint()
  fx_gm.recompile()
  return fx_gm, fx_inputs


def _normalize_minified_fx_gm(
    fx_gm: torch.fx.GraphModule, inputs: Tuple[torch.Tensor]
):
  fx_gm, inputs = _erase_unused_inputs(fx_gm, inputs)
  fx_gm = _lift_dead_ops_to_outputs(fx_gm)
  fx_gm, _ = aot_autograd.aot_export_module(fx_gm, inputs, trace_joint=False)
  fx_gm.__class__.__name__ = _CULPRIT_GRAPH_MODULE_NAME
  return fx_gm, inputs


def _fx_minifier_checker(fx_gm, inputs, runtime_errors=False):
  fx_gm, inputs = _normalize_minified_fx_gm(fx_gm, inputs)

  trivial_aten_ops = {
      torch.ops.aten.view,
      torch.ops.aten.view.default,
  }
  if all(
      node.op in ("placeholder", "output") or node.target in trivial_aten_ops
      for node in fx_gm.graph.nodes
  ):
    return False

  try:
    edge_model = ai_edge_torch.convert(fx_gm.eval(), inputs)
    if runtime_errors:
      edge_model(*inputs)
  except Exception as err:
    return True
  return False


def _search_model(
    predicate_f: Callable[[torch.fx.GraphModule, List[Any]], bool],
    model: Union[torch.export.ExportedProgram, torch.nn.Module],
    export_args: Tuple[Any] = None,
    *,
    max_granularity: Optional[int] = None,
    enable_fx_minifier_logging: bool = False,
) -> Generator[SearchResult, None, None]:
  """Finds subgraphs in the torch model that satify a certain predicate function provided by the users.

  Args:
    predicate_f: a predicate function the users specify. It takes a FX
      (sub)graph and the inputs to this graph, return True if the graph
      satisfies the predicate, return False otherwise.
    model: model in which to search subgraph.
    export_args: A set of args to trace the model with, i.e. model(*args) must
      run. max_granularity - FX minifier arg. The maximum granularity (number of
      nodes) in the returned ATen FX subgraph of the culprit.
    enable_fx_minifier_logging: If true, allows the underlying FX minifier to
      log the progress.
  """

  if isinstance(model, torch.nn.Module):
    try:
      ep = torch.export.export(model, export_args)
    except Exception as err:
      raise ValueError(
          "Your model is not exportable by torch.export.export. Please modify"
          " your model to be torch-exportable first."
      ) from err
  else:
    ep = model

  fx_gm, fx_inputs = utils.exported_program_to_fx_graph_module_and_inputs(ep)
  fx_gm = _normalize_getitem_nodes(fx_gm)

  # HACK: temporarily disable XLA_HLO_DEBUG and create_minified_hlo_graph so that
  # fx_minifier won't dump intermediate stablehlo files to storage.
  # https://github.com/pytorch/pytorch/blob/main/torch/_functorch/fx_minifier.py#L440
  @contextlib.contextmanager
  def disable_minifier_xla_debug():
    xla_hlo_debug_value = None
    if "XLA_HLO_DEBUG" in os.environ:
      xla_hlo_debug_value = os.environ["XLA_HLO_DEBUG"]
      del os.environ["XLA_HLO_DEBUG"]

    create_minified_hlo_graph = (
        torch._functorch.fx_minifier.create_minified_hlo_graph
    )
    torch._functorch.fx_minifier.create_minified_hlo_graph = (
        lambda *args, **kwargs: None
    )

    try:
      yield
    finally:
      if xla_hlo_debug_value is not None:
        os.environ["XLA_HLO_DEBUG"] = xla_hlo_debug_value

      torch._functorch.fx_minifier.create_minified_hlo_graph = (
          create_minified_hlo_graph
      )

  found_culprits_num = 0
  while True:
    try:
      with disable_minifier_xla_debug(), open(os.devnull, "w") as devnull:
        with contextlib.nullcontext() if enable_fx_minifier_logging else utils.redirect_stdio(
            stdout=devnull,
            stderr=devnull,
        ):
          raw_min_fx_gm, raw_min_inputs = fx_minifier(
              fx_gm,
              fx_inputs,
              predicate_f,
              max_granularity=max_granularity,
          )

      min_fx_gm, min_inputs = _normalize_minified_fx_gm(
          raw_min_fx_gm, raw_min_inputs
      )
      found_culprits_num += 1
      yield SearchResult(min_fx_gm, min_inputs)

      fx_gm, fx_inputs = _erase_sub_gm_from_gm(
          fx_gm, fx_inputs, raw_min_fx_gm, raw_min_inputs
      )

    except RuntimeError as e:
      if (
          str(e) == "Input graph did not fail the tester"
          and found_culprits_num > 0
      ):
        break
      raise e


def find_culprits(
    torch_model: torch.nn.Module,
    args: Tuple[Any],
    max_granularity: Optional[int] = None,
    runtime_errors: bool = False,
    *,
    enable_fx_minifier_logging: bool = False,
) -> Generator[Culprit, None, None]:
  """Finds culprits in the AI Edge Torch model conversion.

  Args:
    torch_model: model to export and save
    args: A set of args to trace the model with, i.e. torch_model(*args) must
      run max_granularity - FX minifier arg. The maximum granularity (number of
      nodes) in the returned ATen FX subgraph of the culprit.
    runtime_errors: If true, find culprits for Python runtime errors with
      converted model.
    enable_fx_minifier_logging: If true, allows the underlying FX minifier to
      log the progress.
  """

  fx_minifier_checker = functools.partial(
      _fx_minifier_checker, runtime_errors=runtime_errors
  )

  for search_result in _search_model(
      fx_minifier_checker,
      torch_model,
      args,
      max_granularity=max_granularity,
      enable_fx_minifier_logging=enable_fx_minifier_logging,
  ):
    yield Culprit(
        search_result.graph_module,
        search_result.inputs,
        _runtime_errors=runtime_errors,
    )
