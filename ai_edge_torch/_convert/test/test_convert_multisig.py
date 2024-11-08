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
from torch import nn

from absl.testing import absltest as googletest


class FullyConnectedModel(nn.Module):
  """A simple fully connected model with two fully connected layers."""

  def __init__(self, input_size, hidden_size, output_size):
    super(FullyConnectedModel, self).__init__()
    self.fc = nn.Linear(input_size, hidden_size)  # Fully connected layer
    self.relu = nn.ReLU()  # Activation function
    self.output = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = self.fc(x)
    x = self.relu(x)
    x = self.output(x)
    return x


class FullyConvModel(nn.Module):
  """A simple fully convolutional model with two convolutions."""

  def __init__(self):
    super(FullyConvModel, self).__init__()
    self.conv1 = nn.Conv2d(
        3, 16, kernel_size=3, padding=1
    )  # Input channels: 3 (RGB), Output channels: 16
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(
        16, 1, kernel_size=1
    )  # Output channels: 1 (single channel output)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    return x


class TestConvertMultiSignature(googletest.TestCase):
  """Tests conversion of various modules through multi-signature conversion."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_convert_with_default(self):
    """Tests conversion of a model with two signatures one of which is the default."""
    torch_module = FullyConvModel().eval()

    args = (torch.randn(4, 3, 12, 12),)
    large_args = (torch.randn(4, 3, 24, 24),)

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

  def test_convert_no_default(self):
    """Tests conversion of a model with two signatures none of which is the default."""
    torch_module = FullyConvModel().eval()

    args = (torch.randn(4, 3, 12, 12),)
    large_args = (torch.randn(4, 3, 24, 24),)

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

  def test_convert_signature_helper(self):
    """Tests the ai_edge_torch.signature helper function works."""
    torch_module = FullyConvModel().eval()

    args = (torch.randn(4, 3, 12, 12),)
    large_args = (torch.randn(4, 3, 24, 24),)

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
    fully_conv = FullyConvModel().eval()
    fully_connected = FullyConnectedModel(10, 5, 10).eval()

    fully_conv_args = (torch.randn(4, 3, 12, 12),)
    fully_connected_args = (torch.randn(10),)

    fully_conv_signature_name = "fully_conv"
    fully_connected_signature_name = "fully_connected"

    edge_model = (
        ai_edge_torch.signature(
            fully_conv_signature_name, fully_conv, fully_conv_args
        )
        .signature(
            fully_connected_signature_name,
            fully_connected,
            fully_connected_args,
        )
        .convert(fully_connected, fully_connected_args)
    )

    fully_conv_inference_args = (torch.randn(4, 3, 12, 12),)
    fully_connected_inference_args = (torch.randn(10),)
    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            fully_conv,
            fully_conv_inference_args,
            signature_name=fully_conv_signature_name,
        )
    )
    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            fully_connected,
            fully_connected_inference_args,
            signature_name=fully_connected_signature_name,
        )
    )


if __name__ == "__main__":
  googletest.main()
