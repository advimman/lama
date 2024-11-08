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

"""Common utility functions to verify the reauthored models."""

import logging
from typing import List

from ai_edge_torch.generative.layers import kv_cache as kv_utils
import torch


class ModelWrapper(torch.nn.Module):
  """A wrapper for the model to be verified.

  It unifies the interface of forward() and generate() of models for the
  verification to call.
  """

  def __init__(self, model: torch.nn.Module):
    """Initializes the wrapper.

    Args:
      model (torch.nn.Module): The model which might have different interfaces
        of forward() and generate(). It could be a model built from HuggingFace
        transformers, a regular PyTorch model, or a model re-authored with
        ai_edge_torch Generative API.
    """
    super().__init__()
    self.model = model

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    """Gets output logits by forwarding the input tokens.

    Args:
      tokens (torch.Tensor): The input tokens to forward. Its dimension is
        expected to be (batch_size=1, kv_cache_max_len).

    Returns:
      The output logits.
    """
    raise NotImplementedError("forward() is not implemented.")

  def generate(
      self, prompts: torch.Tensor, max_new_tokens: int
  ) -> torch.IntTensor:
    """Returns the response token IDs to the given prompts tensor.

    The maximum number of tokens to generate might be set by subclasses.

    Args:
      prompts (torch.Tensor): The input token IDs to generate with. Its shape is
        expected to be (batch_size=1, input_ids_len).
      max_new_tokens (int): The maximum number of response token IDs to
        generate.

    Returns:
      The tensor of response token IDs with shape of (batch_size=1,
      response_ids_len).
    """
    raise NotImplementedError("generate() is not implemented.")


class ReauthoredModelWrapper(ModelWrapper):
  """A wrapper for the model reauthored with ai_edge_torch Generative API."""

  def _init_kv_cache(self):
    """Returns an initialized KV cache."""
    return kv_utils.KVCache.from_model_config(self.model.config)

  def _forward_with_kv_cache(
      self,
      tokens: torch.Tensor,
      kv_cache: kv_utils.KVCache,
  ) -> tuple[torch.Tensor, kv_utils.KVCache]:
    """Forwards the model and updates an external KV cache.

    Args:
      tokens (torch.Tensor): The input tokens to forward.
      kv_cache (KVCache): The KV cache to forward.

    Returns:
      The output logits and the updated KV cache.
    """
    input_pos = torch.arange(0, tokens.shape[1], dtype=torch.int)
    output = self.model.forward(tokens, input_pos, kv_cache)
    return output["logits"], output["kv_cache"]

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    logits, _ = self._forward_with_kv_cache(tokens, self._init_kv_cache())
    return logits

  def generate(
      self, prompts: torch.Tensor, max_new_tokens: int
  ) -> torch.IntTensor:
    input_ids = prompts[0].int().tolist()
    kv_cache = self._init_kv_cache()
    for _ in range(max_new_tokens):
      tokens = torch.tensor([input_ids])
      logits, kv_cache = self._forward_with_kv_cache(tokens, kv_cache)
      generated_token = logits[0][-1].argmax().item()
      input_ids.append(generated_token)
    return torch.tensor([input_ids])


class TokenizerWrapper(torch.nn.Module):
  """A wrapper for the tokenizer used for verification."""

  def __init__(self, tokenizer: torch.nn.Module):
    """Initializes the wrapper.

    Args:
      tokenizer (torch.nn.Module): The tokenizer to wrap.
    """
    super().__init__()
    self.tokenizer = tokenizer

  def encode(self, prompts: str) -> torch.Tensor:
    """Encodes the prompts to token IDs."""
    return self.tokenizer.encode(prompts, return_tensors="pt")

  def decode(self, token_ids: torch.Tensor) -> str:
    """Decodes the token IDs to a string."""
    return self.tokenizer.decode(token_ids)


def verify_with_input_ids(
    original_model: ModelWrapper,
    reauthored_model: ReauthoredModelWrapper,
    input_ids: List[int],
    kv_cache_max_len: int = 1024,
    rtol: float = 1e-05,
    atol: float = 1e-05,
) -> bool:
  """Verifies if the model reauthored generates the same output of the oringal.

  It compares only one outputs from the original and the reauthored model.

  Args:
    original_model (ModelWrapper): The original model.
    reauthored_model (ReauthoredModelWrapper): The model reauthored with
      ai_edge_torch Generative API.
    input_ids (List[int]): The input token IDs to forward with.
    kv_cache_max_len (int): The maximum sequence length of the KV cache.
    rtol (float): The relative tolerance for the comparison.
    atol (float): The absolute tolerance for the comparison.

  Returns:
    True if the model reauthored generates the same output of the original.
  """
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.int, device="cpu")
  tokens[0, : len(input_ids)] = torch.tensor([input_ids]).int()

  logging.info("Forwarding the original model...")
  outputs_original = original_model.forward(tokens)
  logits_original = outputs_original[0, len(input_ids) - 1, :]
  logging.info("logits_original: %s", logits_original)

  logging.info("Forwarding the reauthored model...")
  outputs_reauthored = reauthored_model.forward(tokens)
  logits_reauthored = outputs_reauthored[0, len(input_ids) - 1, :]
  logging.info("logits_reauthored: %s", logits_reauthored)

  return torch.allclose(
      logits_original, logits_reauthored, rtol=rtol, atol=atol
  )


def verify_model_with_prompts(
    original_model: ModelWrapper,
    reauthored_model: ReauthoredModelWrapper,
    tokenizer: TokenizerWrapper,
    prompts: str,
    max_new_tokens: int,
) -> bool:
  """Verifies if the model reauthored generates the same answer of the oringal.

  It compares an answer, i.e. multiple continuous outputs generated by the
  original and the reauthored model.

  Args:
    original_model (ModelWrapper): The original model.
    reauthored_model (ReauthoredModelWrapper): The model reauthored with
      ai_edge_torch Generative API.
    tokenizer (TokenizerWrapper): The tokenizer.
    prompts (str): The input prompts to generate answers.
    max_new_tokens (int): The maximum number of new tokens to generate.

  Returns:
    True if the model reauthored generates the same answer of the original.
  """
  prompt_tokens = tokenizer.encode(prompts)

  logging.info("Generating answer with the original model...")
  outputs_original = original_model.generate(prompt_tokens, max_new_tokens)
  response_original = tokenizer.decode(outputs_original[0])
  logging.info("outputs_from_original_model: [[%s]]", response_original)

  logging.info("Generating answer with the reauthored model...")
  outputs_reauthored = reauthored_model.generate(prompt_tokens, max_new_tokens)
  response_reauthored = tokenizer.decode(outputs_reauthored[0])
  logging.info("outputs from reauthored model: [[%s]]", response_reauthored)

  return response_original == response_reauthored


def verify_reauthored_model(
    original_model: ModelWrapper,
    reauthored_model: ReauthoredModelWrapper,
    tokenizer: TokenizerWrapper,
    generate_prompts: List[str],
    max_new_tokens: int = 30,
    forward_input_ids: List[List[int]] = [[1, 2, 3, 4]],
    rtol: float = 1e-05,
    atol: float = 1e-05,
):
  """Verifies the reauthored model against the original model.

  It verifies the reauthored model with two methods:
  1. It compares the output of the original and the reauthored model with an
     arbitrary input.
  2. It compares the answer generated by the original and the reauthored model
     with a prompt.

  It prints out "PASS" or "FAILED" to the console.

  Args:
    original_model (ModelWrapper): The original model.
    reauthored_model (ReauthoredModelWrapper): The model reauthored with
      ai_edge_torch Generative API.
    tokenizer (TokenizerWrapper): The tokenizer.
    generate_prompts (List[str]): List of the input prompts to generate answers.
    max_new_tokens (int): The maximum number of new tokens to generate.
    forward_input_ids (List[torch.Tensor]): List if ihe input token IDs to
      forward with.
    rtol (float): The relative tolerance for the comparison.
    atol (float): The absolute tolerance for the comparison.
  """
  for input_ids in forward_input_ids:
    logging.info("Verifying the reauthored model with input IDs: %s", input_ids)
    if verify_with_input_ids(
        original_model, reauthored_model, input_ids, rtol=rtol, atol=atol
    ):
      logging.info("PASS")
    else:
      logging.error("FAILED")

  for prompts in generate_prompts:
    logging.info("Verifying the reauthored model with prompts:%s", prompts)
    if verify_model_with_prompts(
        original_model, reauthored_model, tokenizer, prompts, max_new_tokens
    ):
      logging.info("PASS")
    else:
      logging.error("FAILED")
