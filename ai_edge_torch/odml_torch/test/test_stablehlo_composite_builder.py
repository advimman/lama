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
import math
import re

from ai_edge_torch import odml_torch
from ai_edge_torch.odml_torch import composite
import torch
import torch.nn.functional as F

from absl.testing import absltest as googletest


def _export_stablehlo_mlir(model, args):
  ep = torch.export.export(model, args)
  mlir = odml_torch.export.exported_program_to_mlir(ep)
  return mlir.get_text()


def _extract_backend_configs(mlir):
  mlir = mlir.replace("\\22", '"')
  configs = []
  for match in re.finditer(r"backend_config\s*=\s*\"(\{.*\})\"", mlir):
    configs.append(match.group(1))
  return "\n".join(configs)


class TestStableHLOCompositeBuilder(googletest.TestCase):
  """Test cases for StableHLOCompositeBuilder.

  This tests the functionality of emitting mark_tensor ops at the boundaries of
  a composite. The actual transformation to a composite happens later in the
  tflite converter.
  """

  def test_build_composite(self):
    class SampleModel(torch.nn.Module):

      def forward(self, x):
        builder = composite.StableHLOCompositeBuilder(name="test.plus_two")
        y = x + 1
        y = builder.mark_inputs(y)
        z = y + 2
        z = builder.mark_outputs(z)
        return z

    mlir = _export_stablehlo_mlir(SampleModel().eval(), (torch.rand((2, 2)),))
    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 2)

  def test_build_multiple_composites(self):
    class SampleModel(torch.nn.Module):

      def plus_one(self, x: torch.Tensor):
        builder = composite.StableHLOCompositeBuilder("test.plus_one")
        x = builder.mark_inputs(x)
        y = x + 1
        y = builder.mark_outputs(y)
        return y

      def plus_two(self, x: torch.Tensor):
        builder = composite.StableHLOCompositeBuilder("test.plus_two")
        x = builder.mark_inputs(x)
        y = x + 2
        y = builder.mark_outputs(y)
        return y

      def forward(self, x):
        x = self.plus_two(x)
        x = x + 3
        x = self.plus_one(x)
        x = x + 4
        x = self.plus_two(x)
        return x

    mlir = _export_stablehlo_mlir(SampleModel().eval(), (torch.rand((2, 2)),))
    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 6)

  def test_build_composite_with_attr(self):
    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def log_softmax(self, x: torch.Tensor, dim: int):
        builder = composite.StableHLOCompositeBuilder(
            name="test.log_softmax", attr={"dim": dim}
        )
        x = builder.mark_inputs(x)
        y = torch.nn.functional.log_softmax(x, dim=dim)
        y = builder.mark_outputs(y)
        return y

      def forward(self, x):
        x = x + 1
        x = self.log_softmax(x, 0)
        x = self.log_softmax(x, 1)
        return x

    mlir = _export_stablehlo_mlir(SampleModel().eval(), (torch.rand((2, 2)),))
    configs_str = _extract_backend_configs(mlir)

    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 4)
    self.assertEqual(configs_str.count('{"dim": 0}'), 1)
    self.assertEqual(configs_str.count('{"dim": 1}'), 1)

  def test_build_composite_with_mix_type_attrs(self):
    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def log_softmax(self, x: torch.Tensor, dim: int):
        builder = composite.StableHLOCompositeBuilder(
            name="test.log_softmax",
            attr={
                "dim": dim,
                "source": "torch.nn",
                "version": 1.0,
            },
        )
        x = builder.mark_inputs(x)
        y = torch.nn.functional.log_softmax(x, dim=dim)
        y = builder.mark_outputs(y)
        return y

      def forward(self, x):
        x = x + 1
        x = self.log_softmax(x, 0)
        return x

    mlir = _export_stablehlo_mlir(SampleModel().eval(), (torch.rand((2, 2)),))
    configs_str = _extract_backend_configs(mlir)

    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 2)
    self.assertEqual(
        configs_str.count('{"dim": 0, "source": "torch.nn", "version": 1.0}'),
        1,
    )

  def test_sdpa_composite(self):
    class SDPAModel(torch.nn.Module):

      def scaled_dot_product_attention(
          self,
          q: torch.Tensor,
          k: torch.Tensor,
          v: torch.Tensor,
          head_size: int,
          mask: torch.Tensor,
      ):
        builder = composite.StableHLOCompositeBuilder(
            "test.scaled_dot_product_attention"
        )
        q, k, v, mask = builder.mark_inputs(q, k, v, mask)

        scale = 1.0 / math.sqrt(head_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=mask is None,
            scale=scale,
        )
        result = y.transpose(1, 2)
        result = builder.mark_outputs(result)
        return result

      def forward(self, q, k, v, mask):
        x = self.scaled_dot_product_attention(
            q,
            k,
            v,
            8,
            mask,
        )
        return x

    query = torch.rand(1, 1, 32, 4)
    key = torch.rand(1, 500, 1, 4)
    value = torch.rand(1, 500, 1, 4)
    mask = torch.rand(1, 1, 1, 500)

    mlir = _export_stablehlo_mlir(
        SDPAModel().eval(),
        (query, key, value, mask),
    )

    # 4 inputs and 1 output
    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 5)

  def test_sdpa_composite_with_attr(self):
    class SDPAModel(torch.nn.Module):

      def scaled_dot_product_attention(
          self,
          q: torch.Tensor,
          k: torch.Tensor,
          v: torch.Tensor,
          head_size: int,
          include_captanh: bool,
      ):
        builder = composite.StableHLOCompositeBuilder(
            name="test.scaled_dot_product_attention",
            attr={"include_captanh": include_captanh},
        )
        q, k, v = builder.mark_inputs(q, k, v)

        scale = 1.0 / math.sqrt(head_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )
        result = y.transpose(1, 2)
        result = builder.mark_outputs(result)
        return result

      def forward(self, q, k, v):
        x = self.scaled_dot_product_attention(q, k, v, 8, True)
        y = self.scaled_dot_product_attention(q, k, v, 8, False)
        return x + y

    query = torch.rand(1, 1, 32, 4)
    key = torch.rand(1, 500, 1, 4)
    value = torch.rand(1, 500, 1, 4)
    mlir = _export_stablehlo_mlir(
        SDPAModel().eval(),
        (query, key, value),
    )

    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 8)

    configs_str = _extract_backend_configs(mlir)
    self.assertEqual(configs_str.count('{"include_captanh": true}'), 1)
    self.assertEqual(
        configs_str.count('{"include_captanh": false}'),
        1,
    )

  def test_build_composite_with_multiple_inputs_outputs(self):
    class SampleModel(torch.nn.Module):

      def mimo_sample(self, a, b, c):
        builder = composite.StableHLOCompositeBuilder(name="test.mimo_sample")

        a, b, c = builder.mark_inputs(a, b, c)
        x = a + b + c
        y = (a - b) * x
        z = (c + 1.0) * a
        x, y, z = builder.mark_outputs(x, y, z)

        result = x + y * z
        return result

      def forward(self, a, b, c):
        x = self.mimo_sample(a, b, c)
        x = self.mimo_sample(a, b, x)
        x = self.mimo_sample(x, x, c)
        return x

    mlir = _export_stablehlo_mlir(
        SampleModel().eval(), (torch.rand(2), torch.rand(2), torch.rand(2))
    )

    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 18)


if __name__ == "__main__":
  googletest.main()
