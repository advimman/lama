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
"""Tests for BuildAtenCompositePass."""

from typing import Callable, Union

from ai_edge_torch import fx_pass_base
from ai_edge_torch import lowertools
from ai_edge_torch._convert import fx_passes
import torch

from absl.testing import absltest as googletest


def _export_to_stablehlo_with_composite(
    func: Union[torch.nn.Module, Callable[..., torch.Tensor]], export_args
):
  """Exports a function to StableHLO with composite ops.

  Args:
    func: A function to export.
    export_args: Arguments to pass to the function.

  Returns:
    A StableHLO program in MLIR text format.
  """
  if not isinstance(func, torch.nn.Module):

    class TestModule(torch.nn.Module):

      def forward(self, *args, **kwargs):
        return func(*args, **kwargs)

    module = TestModule().eval()
  else:
    module = func

  exported_program = torch.export.export(module, export_args)
  exported_program = fx_pass_base.run_passes(
      exported_program,
      [
          fx_passes.BuildAtenCompositePass(),
          fx_passes.CanonicalizePass(),
      ],
  )

  return lowertools.exported_program_to_mlir_text(exported_program)


class TestBuildAtenCompositePass(googletest.TestCase):
  """Tests for BuildAtenCompositePass."""

  def test_hardswish_layer(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.Hardswish()(x), (torch.rand(10, 10),)  # pylint: disable=unnecessary-lambda
    )

    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "aten.hardswish.default"': 1},
        {'stablehlo.custom_call @mark_tensor': 2},
    )

  def test_hardswish_op(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.ops.aten.hardswish.default(x), (torch.rand(10, 10),)  # pylint: disable=unnecessary-lambda
    )

    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "aten.hardswish.default"': 1},
        {'stablehlo.custom_call @mark_tensor': 2},
    )

  def test_avg_pool2d_layer(self):
    """Tests avg_pool2d with a layer."""
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.AvgPool2d(  # pylint: disable=unnecessary-lambda
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[0, 0],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        )(x),
        (torch.rand(1, 3, 6, 6),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "aten.avg_pool2d.default"': 1},
        {'stablehlo.custom_call @mark_tensor': 2},
    )

  def test_avg_pool2d_op(self):
    """Tests avg_pool2d with padding."""
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.avg_pool2d(
            x,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            ceil_mode=False,
            count_include_pad=False,
            divisor_override=None,
        ),
        (torch.rand(1, 3, 6, 6),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "aten.avg_pool2d.default"': 1},
        {'stablehlo.custom_call @mark_tensor': 2},
    )

  def test_avg_pool2d_ceil_mode(self):
    """Tests avg_pool2d with ceil_mode=True."""
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.avg_pool2d(
            x,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            ceil_mode=True,
            count_include_pad=True,
            divisor_override=None,
        ),
        (torch.rand(1, 3, 6, 6),),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "aten.avg_pool2d.default"': 1},
        {'stablehlo.custom_call @mark_tensor': 2},
    )

  def test_gelu_layer(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.GELU()(x), (torch.rand(10, 10),)  # pylint: disable=unnecessary-lambda
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "aten.gelu.default"': 1},
        {'stablehlo.custom_call @mark_tensor': 2},
    )

  def test_approximate_gelu_layer(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.GELU('tanh')(x), (torch.rand(10, 10),)  # pylint: disable=unnecessary-lambda
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "aten.gelu.default"': 1},
        {'stablehlo.custom_call @mark_tensor': 2},
    )

  def test_embedding_lookup_layer(self):
    stablehlo = _export_to_stablehlo_with_composite(
        torch.nn.Embedding(10, 10), (torch.full((1, 10), 0, dtype=torch.long),)
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "odml.embedding_lookup"': 1},
        {'stablehlo.custom_call @mark_tensor': 3},
    )

  def test_embedding_lookup_op(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda *x: torch.ops.aten.embedding.default(*x),
        (torch.rand(10, 10), torch.full((1, 10), 0, dtype=torch.long)),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "odml.embedding_lookup"': 1},
        {'stablehlo.custom_call @mark_tensor': 3},
    )

  def test_embedding_lookup_functional(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda *x: torch.nn.functional.embedding(*x),
        (
            torch.full((1, 10), 0, dtype=torch.long),
            torch.rand(10, 10),
        ),
    )
    lowertools.assert_string_count(
        self,
        stablehlo,
        {'stablehlo.composite "odml.embedding_lookup"': 1},
        {'stablehlo.custom_call @mark_tensor': 3},
    )


if __name__ == '__main__':
  googletest.main()
