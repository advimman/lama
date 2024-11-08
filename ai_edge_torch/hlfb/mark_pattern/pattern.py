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
"""Mark pattern."""

import dataclasses
from typing import Any, Callable, Optional, Union

from ai_edge_torch import fx_pass_base
from ai_edge_torch.hlfb.mark_pattern import passes
import torch
from torch.export.graph_signature import TensorArgument
from torch.fx import Graph
from torch.fx import GraphModule
from torch.fx.passes.utils.matcher_utils import InternalMatch
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher


def _are_equal(x: Any, y: Any) -> bool:
  if type(x) != type(y):
    return False
  if type(x) in [int, str]:
    return x == y
  if isinstance(x, float):
    rel_tol = 1e-07
    abs_tol = 0.0
    return abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
  if isinstance(x, list):
    if len(x) != len(y):
      return False
    return all([_are_equal(a, b) for a, b in zip(x, y)])

  raise Exception(f"Cannot compare type: {type(x)}")


@dataclasses.dataclass
class ScalarAttrTracker:
  """ScalarAttrTracker is used to track the occurrence of a pattern's

  scalar arg/attr in the pattern decomposed graph. Since a scalar attr
  to the pattern can be transformed and turned into a/some ops' scalar
  arg in the decomposed graph, it would be hard to programmatically get
  the attr value from the pattern match. With the tracker and tracking info,
  we could target the position of the decomposed op's scalar arg derived
  from the pattern arg/attr and retrieve the value from the InternalMatch.

  Args:
    name (str): name of the attr to track.
    pattern_arg_pos (int): the index of the attr to track in the pattern's
      export_args.
    transform (Callable): the transform function used when targeting the
      occurrence of the attr value in the decomposed graph. An attr value may be
      transformed during the decomposition and appear as a derived value.
    inverse_transform (Callable): the inverse transform function that maps the
      transformed value back to the original attr value.
  """

  attr_name: str
  pattern_arg_pos: int
  transform: Callable = lambda x: x
  inverse_transform: Callable = lambda x: x
  _source_targets: list[tuple[Any, Any]] = dataclasses.field(
      default_factory=list
  )

  def track(self, *sources):
    """Register magic values to track the (transformed) attr values in

    the pattern decomposed graph.
    """
    for source in sources:
      target = self.transform(source)
      if not _are_equal(self.inverse_transform(target), source):
        raise Exception(
            f"Invalid transform/inverse_transform for {self.attr_name}"
        )
      self._source_targets.append([source, target])
    return self


@dataclasses.dataclass
class ScalarAttrLocation:
  attr_name: str
  node_name: str
  pos: Union[int, str]
  _tracker: ScalarAttrTracker

  @property
  def index(self):
    return self.pos if isinstance(self.pos, int) else None

  @property
  def key(self):
    return self.pos if isinstance(self.pos, str) else None


def _find_scalar_attr(
    pattern_module: torch.nn.Module,
    export_args: tuple[Any],
    tracker: ScalarAttrTracker,
    decomp_table=None,
) -> ScalarAttrLocation:
  scalar_loc_intersections = None
  for source, target in tracker._source_targets:
    track_args = list(export_args)
    track_args[tracker.pattern_arg_pos] = source
    ep = torch.export.export(pattern_module, tuple(track_args))
    if decomp_table is not None:
      ep = fx_pass_base.run_passes(ep, [fx_pass_base.CanonicalizePass()])
      ep = ep.run_decompositions(decomp_table)

    scalar_locs = set()
    nodes = ep.graph_module.graph.nodes
    for n in nodes:
      for arg_pos, arg in enumerate(n.args):
        if type(arg) == type(target) and arg == target:
          scalar_locs.add((n.name, arg_pos))
      for attr, val in n.kwargs.items():
        if type(val) == type(target) and val == target:
          scalar_locs.add((n.name, attr))

    if scalar_loc_intersections is None:
      scalar_loc_intersections = scalar_locs
    else:
      scalar_loc_intersections = scalar_loc_intersections & scalar_locs

    if not scalar_loc_intersections:
      break

  if not scalar_loc_intersections:
    return None
  # Choose any occurrence as the attr provider
  node_name, pos = scalar_loc_intersections.pop()
  return ScalarAttrLocation(tracker.attr_name, node_name, pos, _tracker=tracker)


class Pattern:

  def __init__(
      self,
      name: str,
      module: Union[Callable, torch.nn.Module],
      export_args: tuple[Any],
      *,
      attr_builder: Callable[
          ["Pattern", GraphModule, InternalMatch], Optional[dict[str, Any]]
      ] = None,
      scalar_attr_trackers: list[ScalarAttrTracker] = None,
      decomp_table: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
  ):
    """The PyTorch computation pattern to match against a model.

    Args:
      name (str): the name of the pattern. It would be propagated to the `name`
        attr in StableHLO composite ops for the matched model subgraphs in the
        lowering.
      module (torch.nn.Module or Callable): the PyTorch computation.
      export_args (tuple[Any]): the args used to export the pattern module with
        torch.export.export. If export_args contains non-tensor Python scalars,
        there must be a corresponding attr tracker in `scalar_attr_trackers` for
        each scalar arg. attr_builder (Callable[[Pattern, GraphModule,
        InternalMatch], Optional[dict[str, Any]]]): the callable that produces
        the a scalar attrs dict, which would be propagated to `attr` in
        StableHLO composite ops for the matched model subgraphs in the lowering.
      scalar_attr_trackers (list[ScalarAttrTracker]): the trackers for scalar
        args in `export_args`, which are used to track the attr occurrence(s)
        and retrieve their values from the matched subgraph.
      decomp_table (Optional[dict[torch._ops.OperatorBase, Callable]]): The
        decomposition table to be run on the pattern's exported program.
    """
    if not isinstance(module, torch.nn.Module):

      class PatternModule(torch.nn.Module):

        def __init__(self, func):
          super().__init__()
          self.func = func

        def forward(self, *args, **kwargs):
          return self.func(*args, **kwargs)

      module = PatternModule(module).eval()

    self.name = name
    self.attr_builder = attr_builder
    self._scalar_attr_trackers = (
        scalar_attr_trackers if scalar_attr_trackers else []
    )

    exported_program = torch.export.export(module, export_args)
    if decomp_table is not None:
      exported_program = fx_pass_base.run_passes(
          exported_program, [fx_pass_base.CanonicalizePass()]
      )
      exported_program = exported_program.run_decompositions(decomp_table)

    self.exported_program = exported_program
    self.graph_module = self.exported_program.graph_module

    self._scalar_attr_locations = []
    for tracker in self._scalar_attr_trackers:
      self._scalar_attr_locations.append(
          _find_scalar_attr(
              module, export_args, tracker, decomp_table=decomp_table
          )
      )

    # Sanitize graph_module for more precise pattern matching.
    # The graph_module to match against this pattern should apply equivalent
    # sanitization.
    self.graph_module = passes.remove_clone_ops(self.graph_module)
    self.graph_module = passes.remove_dangling_args(self.graph_module)

    # Builds list of ordered input and output nodes.
    self.graph_nodes_map = {}
    for node in self.graph_module.graph.nodes:
      self.graph_nodes_map[node.name] = node

    self.input_nodes = tuple(
        self.graph_nodes_map[spec.arg.name]
        for spec in self.exported_program.graph_signature.input_specs
        if isinstance(spec.arg, TensorArgument)
    )
    self.output_nodes = tuple(
        self.graph_nodes_map[spec.arg.name]
        for spec in self.exported_program.graph_signature.output_specs
    )

  def register_attr_builder(self, attr_builder):
    self.attr_builder = attr_builder
    return attr_builder

  def match(
      self,
      graph_module: GraphModule,
  ) -> list[tuple[InternalMatch, dict[str, Any]]]:
    matcher = SubgraphMatcher(
        self.graph_module.graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=True,
    )
    matches = matcher.match(graph_module.graph)

    match_with_attrs = []
    # Graph traversal must be done in the reverser order (from SubgraphMatcher).
    for match in matches[::-1]:
      if self.attr_builder is not None:
        attrs = self.attr_builder(self, graph_module, match)
      else:
        attrs = {}

      for loc in self._scalar_attr_locations:
        attrs[loc.attr_name] = self._get_attr_value_from_pattern_match(
            match, loc
        )

      attrs = attrs if attrs else None
      match_with_attrs.append((match, attrs))
    return match_with_attrs

  def _get_attr_value_from_pattern_match(
      self,
      match: InternalMatch,
      loc: ScalarAttrLocation,
  ):
    matched_val = None
    for k, v in match.nodes_map.items():
      if k.name == loc.node_name:
        if loc.index:
          matched_val = v.args[loc.index]
        elif loc.key in v.kwargs.keys():
          matched_val = v.kwargs[loc.key]
    attr_val = loc._tracker.inverse_transform(matched_val)
    return attr_val
