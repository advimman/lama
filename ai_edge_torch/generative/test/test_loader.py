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
# Testing weight loader utilities.

import os
import tempfile

from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.utilities import loader as loading_utils
from ai_edge_torch.generative.utilities import model_builder
import safetensors.torch
import torch

from absl.testing import absltest as googletest


class TestLoader(googletest.TestCase):
  """Unit tests that check weight loader."""

  def test_load_safetensors(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      file_path = os.path.join(temp_dir, "test.safetensors")
      test_data = {"weight": torch.randn(20, 10), "bias": torch.randn(20)}
      safetensors.torch.save_file(test_data, file_path)

      loaded_tensors = loading_utils.load_safetensors(file_path)
      self.assertIn("weight", loaded_tensors)
      self.assertIn("bias", loaded_tensors)

  def test_load_statedict(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      file_path = os.path.join(temp_dir, "test.pt")
      model = torch.nn.Linear(10, 5)
      state_dict = model.state_dict()
      torch.save(state_dict, file_path)

      loaded_tensors = loading_utils.load_pytorch_statedict(file_path)
      self.assertIn("weight", loaded_tensors)
      self.assertIn("bias", loaded_tensors)

  def test_model_loader(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      file_path = os.path.join(temp_dir, "test.safetensors")
      test_weights = {
          "lm_head.weight": torch.randn((32000, 2048)),
          "model.embed_tokens.weight": torch.randn((32000, 2048)),
          "model.layers.0.input_layernorm.weight": torch.randn((2048,)),
          "model.layers.0.mlp.down_proj.weight": torch.randn((2048, 5632)),
          "model.layers.0.mlp.gate_proj.weight": torch.randn((5632, 2048)),
          "model.layers.0.mlp.up_proj.weight": torch.randn((5632, 2048)),
          "model.layers.0.post_attention_layernorm.weight": torch.randn((
              2048,
          )),
          "model.layers.0.self_attn.k_proj.weight": torch.randn((256, 2048)),
          "model.layers.0.self_attn.o_proj.weight": torch.randn((2048, 2048)),
          "model.layers.0.self_attn.q_proj.weight": torch.randn((2048, 2048)),
          "model.layers.0.self_attn.v_proj.weight": torch.randn((256, 2048)),
          "model.norm.weight": torch.randn((2048,)),
      }
      safetensors.torch.save_file(test_weights, file_path)
      cfg = tiny_llama.get_model_config()
      cfg.num_layers = 1
      model = model_builder.DecoderOnlyModel(cfg)

      loader = loading_utils.ModelLoader(file_path, tiny_llama.TENSOR_NAMES)
      # if returns successfully, it means all the tensors were initiallized.
      loader.load(model, strict=True)


if __name__ == "__main__":
  googletest.main()
