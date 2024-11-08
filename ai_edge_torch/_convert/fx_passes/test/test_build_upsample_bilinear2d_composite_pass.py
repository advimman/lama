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
"""Tests for BuildUpsampleBilinear2DCompositePass."""

from typing import Any, Callable, Union

from ai_edge_torch import fx_pass_base
from ai_edge_torch import lowertools
from ai_edge_torch._convert import fx_passes
import torch

from absl.testing import absltest as googletest


def _export_to_stablehlo_with_composite(
    func: Union[torch.nn.Module, Callable[..., Any]], export_args
):
  """Exports a function to StableHLO with BuildInterpolateCompositePass pass."""
  if not isinstance(func, torch.nn.Module):

    class TestModule(torch.nn.Module):

      def forward(self, *args, **kwargs):
        return func(*args, **kwargs)

    module = TestModule().eval()
  else:
    module = func

  exported_program = torch.export.export(module, export_args)
  exported_program = fx_pass_base.run_passes(
      exported_program, [fx_passes.BuildInterpolateCompositePass()]
  )

  return lowertools.exported_program_to_mlir_text(exported_program)


class TestBuildAtenCompositePass(googletest.TestCase):
  """Tests for BuildUpsampleBilinear2DCompositePass."""

  def test_nn_functional_upsample_bilinear(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.upsample(
            x, scale_factor=3.0, mode='bilinear'
        ),
        (torch.rand(1, 3, 10, 10),),
    )

    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "odml.upsample_bilinear2d"': 1,
            'composite_attributes = {align_corners = false, output = dense<30> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"output": [30, 30], "align_corners": false}': 1},
    )

  def test_nn_functional_upsample_bilinear_align_corners(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.upsample(
            x, scale_factor=3.0, mode='bilinear', align_corners=True
        ),
        (torch.rand(1, 3, 10, 10),),
    )

    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "odml.upsample_bilinear2d"': 1,
            'composite_attributes = {align_corners = true, output = dense<30> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"output": [30, 30], "align_corners": true}': 1},
    )

  def test_nn_functional_upsample_bilinear_size(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.upsample(
            x, size=[15, 20], mode='bilinear'
        ),
        (torch.rand(1, 3, 10, 10),),
    )

    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "odml.upsample_bilinear2d"': 1,
            'composite_attributes = {align_corners = false, output = dense<[15, 20]> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"output": [15, 20], "align_corners": false}': 1},
    )

  def test_nn_functional_upsample_bilinear_size_align_corners(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.upsample(
            x, size=[15, 20], mode='bilinear', align_corners=True
        ),
        (torch.rand(1, 3, 10, 10),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "odml.upsample_bilinear2d"': 1,
            'composite_attributes = {align_corners = true, output = dense<[15, 20]> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"output": [15, 20], "align_corners": true}': 1},
    )

  def test_nn_upsample_bilinear(self):
    stablehlo = _export_to_stablehlo_with_composite(
        torch.nn.Upsample(scale_factor=3.0, mode='bilinear').eval(),
        (torch.rand(1, 3, 10, 10),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "odml.upsample_bilinear2d"': 1,
            'composite_attributes = {align_corners = false, output = dense<30> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"output": [30, 30], "align_corners": false}': 1},
    )

  def test_nn_functional_interpolate_bilinear(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.interpolate(
            x, scale_factor=3.0, mode='bilinear'
        ),
        (torch.rand(1, 3, 10, 10),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "odml.upsample_bilinear2d"': 1,
            'composite_attributes = {align_corners = false, output = dense<30> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"output": [30, 30], "align_corners": false}': 1},
    )

  def test_nn_functional_interpolate_bilinear_align_corners(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.interpolate(
            x, scale_factor=3.0, mode='bilinear', align_corners=True
        ),
        (torch.rand(1, 3, 10, 10),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "odml.upsample_bilinear2d"': 1,
            'composite_attributes = {align_corners = true, output = dense<30> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"output": [30, 30], "align_corners": true}': 1},
    )

  def test_nn_functional_interpolate_bilinear_size(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.interpolate(
            x, size=[15, 20], mode='bilinear'
        ),
        (torch.rand(1, 3, 10, 10),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "odml.upsample_bilinear2d"': 1,
            'composite_attributes = {align_corners = false, output = dense<[15, 20]> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"output": [15, 20], "align_corners": false}': 1},
    )

  def test_nn_functional_interpolate_bilinear_size_align_corners(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.interpolate(
            x, size=[15, 20], mode='bilinear', align_corners=True
        ),
        (torch.rand(1, 3, 10, 10),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "odml.upsample_bilinear2d"': 1,
            'composite_attributes = {align_corners = true, output = dense<[15, 20]> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"output": [15, 20], "align_corners": true}': 1},
    )

  def test_nn_functional_interpolate_nearest(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.interpolate(
            x, scale_factor=3.0, mode='nearest'
        ),
        (torch.rand(1, 3, 10, 10),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "tfl.resize_nearest_neighbor"': 1,
            'composite_attributes = {is_nchw_op = true, size = dense<30> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"size": [30, 30], "is_nchw_op": true}': 1},
    )

  def test_nn_functional_interpolate_nearest_size(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.interpolate(
            x, size=[15, 20], mode='nearest'
        ),
        (torch.rand(1, 3, 10, 10),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {
            'stablehlo.composite "tfl.resize_nearest_neighbor"': 1,
            'composite_attributes = {is_nchw_op = true, size = dense<[15, 20]> : tensor<2xi64>}': (
                1
            ),
        },
        {'stablehlo.custom_call @mark_tensor': 2},
        {'{"size": [15, 20], "is_nchw_op": true}': 1},
    )


if __name__ == '__main__':
  googletest.main()
