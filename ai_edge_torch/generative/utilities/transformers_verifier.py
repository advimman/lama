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

"""Utilities for the models predefined in HuggingFace transformers."""

from typing import cast

from ai_edge_torch.generative.utilities import verifier
import torch
import transformers


class TransformersModelWrapper(verifier.ModelWrapper):
  """A wrapper for the model predefined in HuggingFace transformers.

  Verifier expects forward() to return logits while Transformers models return
  an object with `logits` field.

  Transformers models get `max_new_tokens` settings for generate() via
  GenerationConfig.
  """

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    return self.model.forward(tokens).logits

  def generate(
      self, inputs: torch.Tensor, max_new_tokens: int
  ) -> torch.IntTensor:
    gen_config = transformers.GenerationConfig(max_new_tokens=max_new_tokens)
    return self.model.generate(inputs=inputs, generation_config=gen_config)
