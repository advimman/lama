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


import ast
import io
import sys

from ai_edge_torch.debug import find_culprits
import torch

from absl.testing import absltest as googletest

_test_culprit_lib = torch.library.Library("test_culprit", "DEF")

_test_culprit_lib.define("non_lowerable_op(Tensor x) -> Tensor")


@torch.library.impl(
    _test_culprit_lib, "non_lowerable_op", "CompositeExplicitAutograd"
)
def non_lowerable_op(x):
  if x.max() > 10.0:
    return x + 1.0
  return x


@torch.library.impl(_test_culprit_lib, "non_lowerable_op", "Meta")
def non_lowerable_op_meta(x):
  return torch.empty_like(x)


class BadModel(torch.nn.Module):

  def forward(self, x):
    x = x + 1
    x = torch.ops.test_culprit.non_lowerable_op.default(x)
    return x


class TestCulprit(googletest.TestCase):

  def test_find_culprits(self):
    model = BadModel().eval()
    args = (torch.rand(10),)

    culprits = list(find_culprits(model, args))
    self.assertEqual(len(culprits), 1)
    self.assertIn(
        torch.ops.test_culprit.non_lowerable_op.default,
        [n.target for n in culprits[0].graph.nodes],
    )

  def test_valid_culprit_readable(self):
    model = BadModel().eval()
    args = (torch.rand(10),)

    culprits = list(find_culprits(model, args))
    self.assertEqual(len(culprits), 1)

    code = culprits[0].print_readable(print_output=False)

    # The code should be a valid Python code
    ast.parse(code)

  def test_valid_culprit_code(self):
    model = BadModel().eval()
    args = (torch.rand(10),)

    culprits = list(find_culprits(model, args))
    self.assertEqual(len(culprits), 1)

    code = culprits[0].print_code(print_output=False)

    # The code should be a valid Python code
    ast.parse(code)

  def test_find_multiple_culprits(self):
    class MultiBadOpsModel(torch.nn.Module):

      def forward(self, x):
        x = x + 1
        a = torch.ops.test_culprit.non_lowerable_op.default(x)
        b = torch.ops.test_culprit.non_lowerable_op.default(x)
        c = a + b
        d = torch.ops.test_culprit.non_lowerable_op.default(c)
        return d

    model = MultiBadOpsModel().eval()
    args = (torch.rand(10),)

    culprits = list(find_culprits(model, args))
    self.assertEqual(len(culprits), 3)
    for culprit in culprits:
      self.assertIn(
          torch.ops.test_culprit.non_lowerable_op.default,
          [n.target for n in culprit.graph.nodes],
      )

  def test_find_culprits_with_trivial_inputs_outputs(self):

    class MultiBadOpsModel(torch.nn.Module):

      def forward(self, x, y, z):
        x = x + 1
        a = torch.ops.test_culprit.non_lowerable_op.default(x)
        b = torch.ops.test_culprit.non_lowerable_op.default(y)
        return a, b, x, y, a, b

    model = MultiBadOpsModel().eval()
    args = (torch.rand(10), torch.rand(10), torch.rand(10))

    culprits = list(find_culprits(model, args))
    self.assertEqual(len(culprits), 2)
    for culprit in culprits:
      self.assertIn(
          torch.ops.test_culprit.non_lowerable_op.default,
          [n.target for n in culprit.graph.nodes],
      )


if __name__ == "__main__":
  googletest.main()
