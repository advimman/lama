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
"""Tests for search_model."""

from ai_edge_torch.debug import _search_model
import torch

from absl.testing import absltest as googletest


class TestSearchModel(googletest.TestCase):

  def test_search_model_with_ops(self):
    class MultipleOpsModel(torch.nn.Module):

      def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sub_0 = x - 1
        add_0 = y + 1
        mul_0 = x * y
        add_1 = sub_0 + add_0
        mul_1 = add_0 * mul_0
        sub_1 = add_1 - mul_1
        return sub_1

    model = MultipleOpsModel().eval()
    args = (torch.rand(10), torch.rand(10))

    def find_subgraph_with_sub(fx_gm, inputs):
      return torch.ops.aten.sub.Tensor in [n.target for n in fx_gm.graph.nodes]

    results = list(_search_model(find_subgraph_with_sub, model, args))
    self.assertEqual(len(results), 2)
    self.assertIn(
        torch.ops.aten.sub.Tensor, [n.target for n in results[0].graph.nodes]
    )


if __name__ == "__main__":
  googletest.main()
