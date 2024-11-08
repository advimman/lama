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
"""Tests for StableHLOCompositeBuilder."""

import math

from ai_edge_torch import config
from ai_edge_torch import lowertools
from ai_edge_torch.hlfb import StableHLOCompositeBuilder
import torch
import torch.nn.functional as F

from absl.testing import absltest as googletest


def _export_stablehlo_mlir(model, args):
  ep = torch.export.export(model, args)
  return lowertools.exported_program_to_mlir_text(ep)


@googletest.skipIf(
    not config.Config.use_torch_xla,
    reason="The odml_torch counter part is in odml_torch.",
)
class TestStableHLOCompositeBuilder(googletest.TestCase):

  def test_build_composite(self):
    class SampleModel(torch.nn.Module):

      def forward(self, x):
        builder = StableHLOCompositeBuilder(name="test.plus_two")
        y = x + 1
        y = builder.mark_inputs(y)
        z = y + 2
        z = builder.mark_outputs(z)
        return z

    mlir = _export_stablehlo_mlir(SampleModel().eval(), (torch.rand((2, 2)),))
    self.assertEqual(mlir.count('stablehlo.composite "test.plus_two"'), 1)

  def test_build_multiple_composites(self):
    class SampleModel(torch.nn.Module):

      def plus_one(self, x: torch.Tensor):
        builder = StableHLOCompositeBuilder("test.plus_one")
        x = builder.mark_inputs(x)
        y = x + 1
        y = builder.mark_outputs(y)
        return y

      def plus_two(self, x: torch.Tensor):
        builder = StableHLOCompositeBuilder("test.plus_two")
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
    self.assertEqual(mlir.count('stablehlo.composite "test.plus_one"'), 1)
    self.assertEqual(mlir.count('stablehlo.composite "test.plus_two"'), 2)

  def test_build_composite_with_attr(self):
    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def log_softmax(self, x: torch.Tensor, dim: int):
        builder = StableHLOCompositeBuilder(
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
    self.assertEqual(mlir.count('stablehlo.composite "test.log_softmax"'), 2)
    self.assertEqual(mlir.count("composite_attributes = {dim = 0 : i64}"), 1)
    self.assertEqual(mlir.count("composite_attributes = {dim = 1 : i64}"), 1)

  def test_build_composite_with_mix_type_attrs(self):
    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def log_softmax(self, x: torch.Tensor, dim: int):
        builder = StableHLOCompositeBuilder(
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
    self.assertEqual(mlir.count('stablehlo.composite "test.log_softmax"'), 1)
    self.assertEqual(
        mlir.count(
            'composite_attributes = {dim = 0 : i64, source = "torch.nn",'
            " version = 1.000000e+00 : f32}"
        ),
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
        builder = StableHLOCompositeBuilder("test.scaled_dot_product_attention")
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
    self.assertEqual(
        mlir.count('stablehlo.composite "test.scaled_dot_product_attention"'), 1
    )

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
        builder = StableHLOCompositeBuilder(
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
    self.assertEqual(
        mlir.count('stablehlo.composite "test.scaled_dot_product_attention"'), 2
    )
    self.assertEqual(
        mlir.count("composite_attributes = {include_captanh = true}"), 1
    )
    self.assertEqual(
        mlir.count("composite_attributes = {include_captanh = false}"), 1
    )

  def test_build_composite_with_multiple_inputs_outputs(self):
    class SampleModel(torch.nn.Module):

      def mimo_sample(self, a, b, c):
        builder = StableHLOCompositeBuilder(name="test.mimo_sample")

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
    self.assertEqual(mlir.count('stablehlo.composite "test.mimo_sample"'), 3)


if __name__ == "__main__":
  googletest.main()
