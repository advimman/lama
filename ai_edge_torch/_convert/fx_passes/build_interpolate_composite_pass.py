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
"""Build interpolate composite pass."""

import functools

from ai_edge_torch import fx_pass_base
from ai_edge_torch.hlfb import mark_pattern
from ai_edge_torch.hlfb.mark_pattern import pattern as pattern_module
import torch

# For torch nightly released after mid June 2024,
# torch.nn.functional.interpolate no longer gets exported into decomposed graph
# but a single aten op:
# torch.ops.aten.upsample_nearest2d.vec/torch.ops.aten.upsample_bilinear2d.vec.
# This would interefere with our pattern matching based composite builder.
# Here we register the now missing decompositions first.
_INTERPOLATE_DECOMPOSITIONS = torch._decomp.get_decompositions([
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.upsample_nearest2d.vec,
])


@functools.cache
def _get_upsample_bilinear2d_pattern():
  pattern = pattern_module.Pattern(
      "odml.upsample_bilinear2d",
      lambda x: torch.nn.functional.interpolate(
          x, scale_factor=2, mode="bilinear", align_corners=False
      ),
      export_args=(torch.rand(1, 3, 100, 100),),
      decomp_table=_INTERPOLATE_DECOMPOSITIONS,
  )

  @pattern.register_attr_builder
  def attr_builder(pattern, graph_module, internal_match):
    output = internal_match.returning_nodes[0]
    output_h, output_w = output.meta["val"].shape[-2:]
    return {
        "output": (int(output_h), int(output_w)),
        "align_corners": False,
    }

  return pattern


@functools.cache
def _get_upsample_bilinear2d_align_corners_pattern():
  pattern = pattern_module.Pattern(
      "odml.upsample_bilinear2d",
      lambda x: torch.nn.functional.interpolate(
          x, scale_factor=2, mode="bilinear", align_corners=True
      ),
      export_args=(torch.rand(1, 3, 100, 100),),
      decomp_table=_INTERPOLATE_DECOMPOSITIONS,
  )

  @pattern.register_attr_builder
  def attr_builder(graph_module, pattern, internal_match):
    output = internal_match.returning_nodes[0]
    output_h, output_w = output.meta["val"].shape[-2:]
    return {
        "output": (int(output_h), int(output_w)),
        "align_corners": True,
    }

  return pattern


@functools.cache
def _get_interpolate_nearest2d_pattern():
  pattern = pattern_module.Pattern(
      "tfl.resize_nearest_neighbor",
      lambda x: torch.nn.functional.interpolate(
          x, scale_factor=2, mode="nearest"
      ),
      export_args=(torch.rand(1, 3, 100, 100),),
      decomp_table=_INTERPOLATE_DECOMPOSITIONS,
  )

  @pattern.register_attr_builder
  def attr_builder(pattern, graph_module, internal_match):
    output = internal_match.returning_nodes[0]
    output_h, output_w = output.meta["val"].shape[-2:]
    return {
        "size": (int(output_h), int(output_w)),
        "is_nchw_op": True,
    }

  return pattern


class BuildInterpolateCompositePass(fx_pass_base.ExportedProgramPassBase):

  def __init__(self):
    super().__init__()
    self._patterns = [
        _get_upsample_bilinear2d_pattern(),
        _get_upsample_bilinear2d_align_corners_pattern(),
        _get_interpolate_nearest2d_pattern(),
    ]

  def call(self, exported_program: torch.export.ExportedProgram):
    exported_program = fx_pass_base.run_passes(
        exported_program, [fx_pass_base.CanonicalizePass()]
    )
    exported_program = exported_program.run_decompositions(
        _INTERPOLATE_DECOMPOSITIONS
    )

    graph_module = exported_program.graph_module
    for pattern in self._patterns:
      graph_module = mark_pattern.mark_pattern(graph_module, pattern)

    graph_module.graph.lint()
    graph_module.recompile()
    return fx_pass_base.ExportedProgramPassResult(exported_program, True)
