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
"""Tests for OptimizeLayoutTransposesPass."""

from typing import Callable, Union

from ai_edge_torch import fx_pass_base
from ai_edge_torch._convert import fx_passes
import torch
import torch.utils._pytree as pytree
import torchvision

from absl.testing import absltest as googletest


def export_with_pass(
    func: Union[torch.nn.Module, Callable[..., torch.Tensor]],
    export_args: list[torch.Tensor],
) -> torch.export.ExportedProgram:
  """Exports a function with OptimizeLayoutTransposesPass."""
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
          fx_passes.OptimizeLayoutTransposesPass(),
          fx_passes.CanonicalizePass(),
      ],
  )
  return exported_program


class TestOptimizeLayoutTransposesPass(googletest.TestCase):
  """Tests for OptimizeLayoutTransposesPass."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def assert_outputs_allclose(self, m1, m2, args):
    out1 = m1(*args)
    out2 = m2(*args)
    out1, _ = pytree.tree_flatten(out1)
    out2, _ = pytree.tree_flatten(out2)
    self.assertEqual(len(out1), len(out2))
    for o1, o2 in zip(out1, out2):
      self.assertTrue(torch.allclose(o1, o2, atol=1e-5))

  def test_torchvision_mobilenet_v3_small(self):
    model = torchvision.models.mobilenet_v3_small().eval()
    forward_args = lambda: (torch.rand(1, 3, 224, 224),)

    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(
        model, exported_program.module(), forward_args()
    )

  def test_torchvision_resnet18(self):
    model = torchvision.models.resnet18().eval()
    forward_args = lambda: (torch.rand(1, 3, 224, 224),)

    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(
        model, exported_program.module(), forward_args()
    )

  def test_native_group_norm_no_weight_bias(self):
    batch_size = 16
    num_channels = 640
    flattened_inner_size = 256
    num_groups = 32
    eps = 1e-6

    class SampleModel(torch.nn.Module):

      def forward(self, x):
        x = torch.nn.AvgPool2d(2)(x)
        x = torch.ops.aten.native_group_norm(
            x,
            None,
            None,
            batch_size,
            num_channels,
            flattened_inner_size,
            num_groups,
            eps,
        )[0]
        x = torch.nn.AvgPool2d(2)(x)
        return x

    model = SampleModel().eval()
    forward_args = lambda: (torch.rand(16, 640, 32, 32) * 1000,)
    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(
        model, exported_program.module(), forward_args()
    )

  def test_native_group_norm_large_weight_bias(self):
    batch_size = 16
    num_channels = 640
    flattened_inner_size = 256
    num_groups = 32
    eps = 1e-6

    class SampleModel(torch.nn.Module):

      def forward(self, x, weight, bias):
        x = torch.nn.AvgPool2d(2)(x)
        x = torch.ops.aten.native_group_norm(
            x,
            weight,
            bias,
            batch_size,
            num_channels,
            flattened_inner_size,
            num_groups,
            eps,
        )[0]
        x = torch.nn.AvgPool2d(2)(x)
        return x

    model = SampleModel().eval()
    forward_args = lambda: (
        torch.rand(16, 640, 32, 32) * 1000,
        torch.rand([640]) * 1000,
        torch.rand([640]) * 1000,
    )
    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(
        model, exported_program.module(), forward_args()
    )


if __name__ == '__main__':
  googletest.main()
