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
from ai_edge_torch import odml_torch
import numpy as np
import torch
import torchvision

from absl.testing import absltest as googletest


class TensorflowIntegrationTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_mlir_lowered_call(self):
    class AddModel(torch.nn.Module):

      def forward(self, x, y):
        return x + y + x + y

    model = AddModel().eval()
    forward_args = lambda: (torch.rand((10, 10)), torch.rand((10, 10)))
    ep = torch.export.export(model, forward_args())

    lowered = odml_torch.export.exported_program_to_mlir(ep)

    args = forward_args()
    torch_output = model(*args).detach().numpy()
    lowering_output = np.array(lowered(*args))

    self.assertTrue(np.allclose(lowering_output, torch_output))

  @googletest.skip("b/353280409")
  def test_dynamic_mlir_lowered_call(self):
    class AddModel(torch.nn.Module):

      def forward(self, x, y):
        return x + y + x + y

    model = AddModel().eval()
    batch = torch.export.Dim("batch")
    ep = torch.export.export(
        model,
        (torch.rand((2, 10)), torch.rand((2, 10))),
        dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
    )

    lowered = odml_torch.export.exported_program_to_mlir(ep)

    val_args = (torch.rand((10, 10)), torch.rand((10, 10)))
    torch_output = model(*val_args).detach().numpy()
    lowering_output = np.array(lowered(*val_args))

    self.assertTrue(np.allclose(lowering_output, torch_output))

  def test_resnet18(self):
    model = torchvision.models.resnet18().eval()
    forward_args = lambda: (torch.rand((1, 3, 224, 224)),)

    ep = torch.export.export(model, forward_args())

    lowered = odml_torch.export.exported_program_to_mlir(ep)

    args = forward_args()
    torch_output = model(*args).detach().numpy()
    lowering_output = np.array(lowered(*args))

    self.assertTrue(np.allclose(lowering_output, torch_output, atol=1e-5))


if __name__ == "__main__":
  googletest.main()
