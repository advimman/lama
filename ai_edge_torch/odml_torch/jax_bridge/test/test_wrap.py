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
"""jax_bridge.wrap related tests."""

from ai_edge_torch import odml_torch
from ai_edge_torch.odml_torch import jax_bridge
import jax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func

from absl.testing import absltest as googletest


class TestWrap(googletest.TestCase):

  def test_wrap(self):
    with (
        odml_torch.export_utils.create_ir_context() as context,
        ir.Location.unknown(),
    ):
      module = ir.Module.create()
      lctx = odml_torch.export.LoweringContext(context, module)

      @jax_bridge.wrap
      def wrapped_add(a: jax.Array, b: jax.Array):
        return a + b

      with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            name="main",
        )
        def _main(a, b):
          return wrapped_add(lctx, a, b)

    ir_text = str(module)
    self.assertIn("%0 = call @wrapped_add_", ir_text)
    self.assertIn(
        "%0 = stablehlo.add %arg0, %arg1 : tensor<10x10xf32>", ir_text
    )

  def test_wrap_with_ir_input_names(self):
    with (
        odml_torch.export_utils.create_ir_context() as context,
        ir.Location.unknown(),
    ):
      module = ir.Module.create()
      lctx = odml_torch.export.LoweringContext(context, module)

      def retb(a: jax.Array, b: jax.Array):
        return b

      wrapped_retb = jax_bridge.wrap(retb, ir_input_names=["b"])

      with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            name="main",
        )
        def _main(a, b):
          return wrapped_retb(lctx, a, b)

    ir_text = str(module)
    self.assertIn("%0 = call @retb_", ir_text)
    self.assertIn("return %arg0 : tensor<10x10xf32>", ir_text)


if __name__ == "__main__":
  googletest.main()
