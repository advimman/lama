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
"""Tests for InjectMlirDebuginfoPass."""

from typing import Callable, Union

from ai_edge_torch import fx_pass_base
from ai_edge_torch import lowertools
from ai_edge_torch._convert import fx_passes
import torch

from absl.testing import absltest as googletest


def _export_to_stablehlo_with_composite(
    func: Union[torch.nn.Module, Callable[..., torch.Tensor]], export_args
):
  """Exports a function to StableHLO with InjectMlirDebuginfoPass."""
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
          fx_passes.InjectMlirDebuginfoPass(),
          fx_passes.CanonicalizePass(),
      ],
  )

  return lowertools.exported_program_to_mlir_text(exported_program)


class TestInjectMlirDebuginfoPass(googletest.TestCase):
  """Tests for InjectMlirDebuginfoPass."""

  def test_write_torch_layers_debuginfo(self):
    """Tests if InjectMlirDebuginfoPass writes torch layers' debuginfo."""

    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()

      def forward(self, x, y):
        z = x + y
        z = self.softmax(z)
        return z

    stablehlo = _export_to_stablehlo_with_composite(
        SampleModel().eval(), (torch.rand(10, 10), torch.rand(10, 10))
    )
    self.assertIn(
        'SampleModel/torch.nn.modules.activation.Softmax_softmax;"', stablehlo
    )
    self.assertIn('SampleModel;"', stablehlo)


if __name__ == '__main__':
  googletest.main()
