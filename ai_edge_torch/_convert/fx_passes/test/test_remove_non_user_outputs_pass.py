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
"""Tests for RemoveNonUserOutputsPass."""

import re
from typing import Any, Callable, Union

from ai_edge_torch import fx_pass_base
from ai_edge_torch import lowertools
from ai_edge_torch._convert import fx_passes
import torch

from absl.testing import absltest as googletest


def _export_to_stablehlo(
    func: Union[torch.nn.Module, Callable[..., Any]], export_args
):
  if not isinstance(func, torch.nn.Module):

    class TestModule(torch.nn.Module):

      def forward(self, *args, **kwargs):
        return func(*args, **kwargs)

    module = TestModule().eval()
  else:
    module = func

  exported_program = torch.export.export(module, export_args)
  exported_program = fx_pass_base.run_passes(
      exported_program, [fx_passes.RemoveNonUserOutputsPass()]
  )

  return lowertools.exported_program_to_mlir_text(exported_program)


class TestRemoveNonUserOutputsPass(googletest.TestCase):
  """Tests for TestRemoveNonUserOutputsPass."""

  def test_remove_input_mutations(self):
    def f(x):
      x += 1
      x = x + 2
      x = x + 3
      return x

    stablehlo = _export_to_stablehlo(
        f,
        (torch.rand(10, 10),),
    )
    first_return = re.search(
        r"^\s+return[^\n]+\n",
        stablehlo,
        re.MULTILINE,
    ).group(0)

    self.assertRegex(first_return.strip(), r"return %\d+ : tensor<10x10xf32>")


if __name__ == "__main__":
  googletest.main()
