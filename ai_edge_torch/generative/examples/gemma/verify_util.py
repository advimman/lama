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

"""Utility functions to verify the reauthored Gemma model."""

import logging
import os
from typing import List, Tuple

import ai_edge_torch.generative.layers.attention_utils as attn_utils
from ai_edge_torch.generative.utilities import verifier
from gemma import config as gemma_config
from gemma import model as gemma_model
import torch


class GemmaWrapper(verifier.ModelWrapper):
  """Gemma model wrapper for verification.

  Verifier calls model.forward() with maxium sequence length (1024) expecting
  the output is logits while Gemma gets the input tokens with the actual length
  and returns logits in a tuple.

  Verifier runs tokenizer before model.generate() while Gemma runs the tokenizer
  inside model.generate().
  """

  def _get_actual_input_len(self, tokens: torch.Tensor) -> int:
    for i in range(tokens.shape[1]):
      if tokens[0, i] == 0:
        return i
    return tokens.shape[1]

  def _get_kv_caches(
      self, max_seq_len: int
  ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    config = self.model.config
    cache_size = (1, max_seq_len, config.num_key_value_heads, config.head_dim)
    cache = torch.zeros(cache_size)
    return [
        (cache.clone(), cache.clone()) for _ in range(config.num_hidden_layers)
    ]

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    """Forwards the model after reducing input tokens to the actual length."""
    actual_input_len = self._get_actual_input_len(tokens)
    input_pos = torch.arange(0, actual_input_len, dtype=torch.long)
    mask_cache = attn_utils.build_causal_mask_cache(tokens.shape[1])
    _, logits = self.model.forward(
        input_token_ids=tokens[0, :actual_input_len].unsqueeze(0),
        input_positions=input_pos,
        kv_write_indices=None,
        kv_caches=self._get_kv_caches(tokens.shape[1]),
        mask=mask_cache.index_select(2, input_pos),
        output_positions=input_pos,
        temperatures=None,
        top_ps=torch.tensor([1.0], dtype=torch.float),
        top_ks=torch.tensor([1], dtype=torch.long),
    )
    return logits

  def generate(
      self, tokens: torch.Tensor, max_new_tokens: int
  ) -> torch.IntTensor:
    """Generates the response after decoding the tokens into a string."""
    prompts = self.model.tokenizer.decode(tokens[0].tolist())
    response = self.model.generate(
        prompts, device="cpu", output_len=max_new_tokens, top_k=1
    )
    return torch.tensor([self.model.tokenizer.encode(prompts + response)])


class GemmaTokenizerWrapper(verifier.TokenizerWrapper):
  """Tokenizer wrapper for verification.

  Verifier expects the tokenizer to handle tokens in torch.Tensor while Gemma
  tokenizer expects tokens in a list.
  """

  def encode(self, text: str, **_) -> torch.Tensor:
    """Adds one more dimension to the output of the tokenizer."""
    return torch.tensor([self.tokenizer.encode(text)])

  def decode(self, tokens: torch.Tensor) -> str:
    """Decodes the token sequence after converting to a list."""
    return self.tokenizer.decode(tokens.tolist())


def verify_reauthored_gemma_model(
    checkpoint: str,
    variant: str,
    reauthored_model: torch.nn.Module,
    generate_prompts: List[str],
    forward_input_ids: List[List[int]],
    weight_filename: str = "model.ckpt",
    tokenizer_filename: str = "tokenizer.model",
    max_new_tokens: int = 20,
    rtol: float = 1e-05,
    atol: float = 1e-05,
):
  """Verifies the reauthored Gemma model against the original model."""
  config = gemma_config.get_model_config(variant)
  config.tokenizer = os.path.join(checkpoint, tokenizer_filename)
  # Use float32 to be compatible with the reauthored model.
  config.dtype = torch.float32

  logging.info("Loading the original model from: %s", checkpoint)
  original_model = gemma_model.GemmaForCausalLM(config).eval()
  original_model.load_weights(os.path.join(checkpoint, weight_filename))

  verifier.verify_reauthored_model(
      original_model=GemmaWrapper(original_model),
      reauthored_model=verifier.ReauthoredModelWrapper(reauthored_model),
      tokenizer=GemmaTokenizerWrapper(original_model.tokenizer),
      generate_prompts=generate_prompts,
      max_new_tokens=max_new_tokens,
      forward_input_ids=forward_input_ids,
      rtol=rtol,
      atol=atol,
  )
