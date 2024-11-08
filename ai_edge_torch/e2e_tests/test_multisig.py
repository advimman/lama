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
"""Tests for multi-signature conversion."""

import ai_edge_torch
from ai_edge_torch.testing import model_coverage
import torch
import torchvision

from absl.testing import absltest as googletest


class TestConvertMultiSignature(googletest.TestCase):
  """Tests conversion of various modules through multi-signature conversion."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_convert_mobilenet_v2_with_default(self):
    """Tests conversion of a model with two signatures one of which is the default."""
    torch_module = torchvision.models.mobilenet_v2().eval()

    args = (torch.randn(4, 3, 224, 224),)
    large_args = (torch.randn(4, 3, 336, 336),)

    signature_name = "large_input"

    edge_model = ai_edge_torch.signature(
        signature_name, torch_module, large_args
    ).convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )
    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, large_args, signature_name=signature_name
        )
    )

  def test_convert_mobilenet_v2_no_default(self):
    """Tests conversion of a model with two signatures none of which is the default."""
    torch_module = torchvision.models.mobilenet_v2().eval()

    args = (torch.randn(4, 3, 224, 224),)
    large_args = (torch.randn(4, 3, 336, 336),)

    signature_name_1 = "input"
    signature_name_2 = "large_input"

    edge_model = (
        ai_edge_torch.signature(signature_name_1, torch_module, args)
        .signature(signature_name_2, torch_module, large_args)
        .convert()
    )

    with self.assertRaises(ValueError):
      edge_model(*args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, args, signature_name=signature_name_1
        )
    )
    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            torch_module,
            large_args,
            signature_name=signature_name_2,
        )
    )

  def test_convert_mobilenet_v2_signature_helper(self):
    """Tests the ai_edge_torch.signature helper function works."""
    torch_module = torchvision.models.mobilenet_v2().eval()

    args = (torch.randn(4, 3, 224, 224),)
    large_args = (torch.randn(4, 3, 336, 336),)

    signature_name = "large_input"

    edge_model = ai_edge_torch.signature(
        signature_name, torch_module, large_args
    ).convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )
    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, large_args, signature_name=signature_name
        )
    )

  def test_convert_separate_modules(self):
    """Tests conversion of two completely different modules as separate signatures."""
    mobilentv2 = torchvision.models.mobilenet_v2().eval()
    resnet18 = torchvision.models.resnet18().eval()

    mobilenet_args = (torch.randn(4, 3, 224, 224),)
    resnet_args = (torch.randn(4, 3, 224, 224),)

    mobilenet_signature_name = "mobilentv2"
    resnet_signature_name = "resnet18"

    edge_model = (
        ai_edge_torch.signature(
            mobilenet_signature_name, mobilentv2, mobilenet_args
        )
        .signature(resnet_signature_name, resnet18, resnet_args)
        .convert(resnet18, resnet_args)
    )

    mobilenet_inference_args = (torch.randn(4, 3, 224, 224),)
    resnet_inference_args = (torch.randn(4, 3, 224, 224),)
    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            mobilentv2,
            mobilenet_inference_args,
            signature_name=mobilenet_signature_name,
        )
    )
    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            resnet18,
            resnet_inference_args,
            signature_name=resnet_signature_name,
        )
    )


if __name__ == "__main__":
  googletest.main()
