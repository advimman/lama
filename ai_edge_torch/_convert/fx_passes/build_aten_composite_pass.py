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

from typing import Any, Callable
from ai_edge_torch import fx_pass_base
from ai_edge_torch import lowertools
import torch
import torch.utils._pytree as pytree

_composite_builders: dict[
    Callable, Callable[[torch.fx.GraphModule, torch.fx.Node], None]
] = {}


def _register_composite_builder(op):
  def inner(func):
    if isinstance(op, torch._ops.OpOverloadPacket):
      for overload in op.overloads():
        _composite_builders[getattr(op, overload)] = func
    else:
      _composite_builders[op] = func
    return func

  return inner


def _tree_map_to_composite_attr_values(
    values, *, stringify_incompatible_values=True
):
  """Convert a tree of values to a tree of composite attribute values.

  This is used for pre-processing op attributes before passing them to
  the composite op as attributes.

  Args:
    values: A tree of values.
    stringify_incompatible_values: If True, stringify values that are not
      compatible with composite attributes.

  Returns:
    A tree of composite attribute values.
  """

  def convert(value):
    nonlocal stringify_incompatible_values
    if value is None:
      return "py_None"
    if isinstance(value, (str, int, float, bool)):
      return value

    if stringify_incompatible_values:
      return str(value)
    return value

  return pytree.tree_map(convert, values)


class TorchOpArgumentsMapper:
  """A helper class to map op arguments to kwargs.

  This is mainly used to extract the default values for op arguments and present
  all arguments as kwargs.
  """

  def __init__(self, op):
    if isinstance(op, torch._ops.OpOverloadPacket):
      op = op.default

    assert hasattr(op, "_schema")
    self.op = op
    self.arg_specs = [
        (spec.name, spec.default_value) for spec in op._schema.arguments
    ]

  def get_full_kwargs(self, args, kwargs=None) -> dict[str, Any]:
    """Extracts all arguments of the op as kwargs.

    Inspect the op's schema and extract all its args and kwargs into one single
    kwargs dict, with default values for those unspecified args and kwargs.

    Args:
      args: The op's arguments.
      kwargs: The op's kwargs.

    Returns:
      A kwargs dict with all args and kwargs.
    """
    full_kwargs = {**(kwargs or {})}

    for arg, (name, _) in zip(args, self.arg_specs):
      full_kwargs[name] = arg

    for name, default_value in self.arg_specs[len(args) :]:
      if name not in full_kwargs:
        full_kwargs[name] = default_value

    return full_kwargs


@_register_composite_builder(torch.ops.aten.hardswish.default)
def _aten_hardswish(_: torch.fx.GraphModule, node: torch.fx.Node):
  """Build a composite for aten.hardswish.default."""
  op = node.target

  def hardswish(self: torch.Tensor):
    nonlocal op
    builder = lowertools.StableHLOCompositeBuilder("aten.hardswish.default")
    self = builder.mark_inputs(self)
    output = op(self)
    output = builder.mark_outputs(output)
    return output

  node.target = hardswish


@_register_composite_builder(torch.ops.aten.gelu.default)
def _aten_gelu(_: torch.fx.GraphModule, node: torch.fx.Node):
  """Build a composite for aten.gelu.default."""
  op = node.target
  args_mapper = TorchOpArgumentsMapper(op)

  def gelu(*args, **kwargs):
    nonlocal op, args_mapper

    full_kwargs = args_mapper.get_full_kwargs(args, kwargs)

    # TFLite supports exact and tanh approximate.
    if (
        full_kwargs["approximate"] != "none"
        and full_kwargs["approximate"] != "tanh"
    ):
      return op(*args, **kwargs)

    builder = lowertools.StableHLOCompositeBuilder(
        "aten.gelu.default",
        attr=_tree_map_to_composite_attr_values({
            "approximate": full_kwargs["approximate"],
        }),
    )
    full_kwargs["self"] = builder.mark_inputs(full_kwargs["self"])
    output = op(full_kwargs["self"])
    output = builder.mark_outputs(output)
    return output

  node.target = gelu


@_register_composite_builder(torch.ops.aten.avg_pool2d.default)
def _aten_avg_pool2d(_: torch.fx.GraphModule, node: torch.fx.Node):
  """Build a composite for aten.avg_pool2d.default."""
  op = node.target
  args_mapper = TorchOpArgumentsMapper(op)

  def avg_pool2d(*args, **kwargs):
    nonlocal op, args_mapper

    full_kwargs = args_mapper.get_full_kwargs(args, kwargs)

    def is_same_padding(
        input_shape: list[int],
        kernel_size: list[int],
        stride: list[int],
        padding: list[int],
    ):
      for dim_input_size, dim_kernel_size, dim_stride, dim_padding in zip(
          input_shape, kernel_size, stride, padding
      ):
        dim_output_size = int((dim_input_size + dim_stride - 1) / dim_stride)
        padding_needed = max(
            0,
            (dim_output_size - 1) * dim_stride
            + dim_kernel_size
            - dim_input_size,
        )
        if padding_needed % 2 != 0:
          return False

        if padding_needed // 2 != dim_padding:
          return False
      return True

    def is_valid_padding(padding: list[int]):
      return not any(padding)

    # We prefer to avoid passing empty arrays to composite attributes
    # as they will be lowered to an ArrayAttr so canonicalizing according
    # to the default behaviour here.
    if not full_kwargs["stride"]:
      full_kwargs["stride"] = full_kwargs["kernel_size"]

    # Only wrap in a composite when the underlying converter can handle it.
    # TODO We should be able to remove this if the converter can inline composites when it can not handle them.

    # We don't cover any cases where the divisor_override is set.
    if full_kwargs["divisor_override"] is not None:
      return op(*args, **kwargs)

    if full_kwargs["ceil_mode"] and not full_kwargs["count_include_pad"]:
      return op(*args, **kwargs)

    # We also can not cover a case where count_include_pad is False but the padding is custom.
    if (
        not full_kwargs["count_include_pad"]
        and not is_valid_padding(full_kwargs["padding"])
        and not is_same_padding(
            list(full_kwargs["self"].shape)[2:],
            full_kwargs["kernel_size"],
            full_kwargs["stride"],
            full_kwargs["padding"],
        )
    ):
      return op(*args, **kwargs)

    builder = lowertools.StableHLOCompositeBuilder(
        "aten.avg_pool2d.default",
        attr=_tree_map_to_composite_attr_values({
            "kernel_size": full_kwargs["kernel_size"],
            "stride": full_kwargs["stride"],
            "padding": full_kwargs["padding"],
            "ceil_mode": full_kwargs["ceil_mode"],
            "count_include_pad": full_kwargs["count_include_pad"],
            "divisor_override": full_kwargs["divisor_override"],
        }),
    )

    full_kwargs["self"] = builder.mark_inputs(full_kwargs["self"])
    output = op(**full_kwargs)
    output = builder.mark_outputs(output)
    return output

  node.target = avg_pool2d


@_register_composite_builder(torch.ops.aten.embedding.default)
def _aten_embedding(gm: torch.fx.GraphModule, node: torch.fx.Node):
  op = node.target
  args_mapper = TorchOpArgumentsMapper(op)

  def embedding(*args, **kwargs):
    nonlocal op, args_mapper
    full_kwargs = args_mapper.get_full_kwargs(args, kwargs)
    _, embedding_dim = full_kwargs["weight"].size()
    idx = full_kwargs["indices"]

    # Explicitly cast to INT32. This places the CastOp outside of the HLFB.
    idx = idx.type(torch.int)
    original_idx_shape = idx.size()

    # Explicitly reshape to 1D. This places the ReshapeOp outside of the HLFB.
    idx = torch.reshape(idx, (idx.numel(),))

    builder = lowertools.StableHLOCompositeBuilder("odml.embedding_lookup")
    full_kwargs["indices"], full_kwargs["weight"] = builder.mark_inputs(
        idx,
        full_kwargs["weight"],
    )
    output = op(**full_kwargs)
    output = builder.mark_outputs(output)

    # Explicitly reshape back to the original shape. This places the ReshapeOp outside of the HLFB.
    output = torch.reshape(output, (*(original_idx_shape), embedding_dim))
    return output

  node.target = embedding


class BuildAtenCompositePass(fx_pass_base.PassBase):

  def call(self, graph_module: torch.fx.GraphModule):
    for node in graph_module.graph.nodes:
      if node.target in _composite_builders:
        _composite_builders[node.target](graph_module, node)

    graph_module.graph.lint()
    graph_module.recompile()
    return fx_pass_base.PassResult(graph_module, True)
