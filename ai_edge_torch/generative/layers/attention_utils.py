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
# Common utility functions used with attention module.

import math
from typing import Tuple

import torch


def build_rope_cache(
    size: int,
    dim: int,
    base: int = 10000,
    condense_ratio: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Precomputes Rotary Positional Embeddings.

  Precompute Rotary Positional Embedding Sin and Cos values for quick lookup
  during the inference.

  Args:
      size (int): The size of the built cache.
      dim (int): Each sequence's dimmension.
      base (int, optional): Rope base value. Defaults to 10000.
      condense_ratio (int, optional): The ratio by which sequence indicies are
        condensed. Defaults to 1.
      dtype (torch.dtype, optional): Output tensor's data type. Defaults to
        torch.float32.
      device (torch.device, optional): Output tensor's data type. Defaults to
        None in which case "cpu" is used.

  Returns:
      Tuple[torch.Tensor, torch.Tensor]: Rope's Cosine and Sine waves.
  """
  if device is None:
    device = torch.device('cpu')
  theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
  seq_idx = torch.arange(size) / condense_ratio
  idx_theta = torch.outer(seq_idx, theta)
  cos = torch.cos(idx_theta).to(dtype=dtype, device=device)
  sin = torch.sin(idx_theta).to(dtype=dtype, device=device)
  return cos, sin


def build_causal_mask_cache(
    size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
  """Build a cache for causal attention mask.

  Args:
      size (int): The size of the built mask cache.
      dtype (torch.dtype, optional): Output tensor's data type. Defaults to
        torch.float32.
      device (torch.device, optional): Output tensor's data type. Defaults to
        None in which case "cpu" is used.

  Returns:
      torch.Tensor: Causal attention mask.
  """

  if device is None:
    device = torch.device('cpu')
  mask = torch.full((size, size), float('-inf'), dtype=dtype, device=device)
  return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)


def build_sliding_window_mask_cache(
    size: int,
    window_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
  """Build a cache for a sliding window mask.

  Args:
      size (int): The size of the built mask cache.
      window_size (int): The window size that is "seen" by a token.
      dtype (torch.dtype, optional): Output tensor's data type. Defaults to
        torch.float32.
      device (torch.device, optional): Output tensor's data type. Defaults to
        None in which case "cpu" is used.

  Returns:
      torch.Tensor: Causal attention mask.
  """

  mask = build_causal_mask_cache(size, dtype, device)
  all_ones = torch.ones_like(mask)
  window_size = min(size, window_size)
  sliding_mask = torch.triu(all_ones, -1 * window_size + 1) * torch.tril(
      all_ones, window_size - 1
  )
  return torch.where(sliding_mask == 1, mask, float('-inf'))


def relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
  """Adapted from Mesh Tensorflow:

  https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

  Translate relative position to a bucket number for relative attention. The
  relative position is defined as
  memory_position - query_position, i.e. the distance in tokens from the
  attending position to the attended-to
  position. If bidirectional=False, then positive relative positions are
  invalid. We use smaller buckets for
  small absolute relative_position and larger buckets for larger absolute
  relative_positions. All relative
  positions >=max_distance map to the same bucket. All relative positions
  <=-max_distance map to the same bucket.
  This should allow for more graceful generalization to longer sequences than
  the model has been trained on

  Args:
      relative_position: an int32 Tensor
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer for number of buckets.
      max_distance: an integer for max distance.

  Returns:
      a Tensor with the same shape as relative_position, containing int32 values
      in the range [0, num_buckets)
  """
  relative_buckets = 0
  if bidirectional:
    num_buckets //= 2
    relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)
  else:
    relative_position = -torch.min(
        relative_position, torch.zeros_like(relative_position)
    )
  # now relative_position is in the range [0, inf)

  # half of the buckets are for exact increments in positions
  max_exact = num_buckets // 2
  is_small = relative_position < max_exact

  # The other half of the buckets are for logarithmically bigger bins in
  # positions up to max_distance
  relative_position_if_large = max_exact + (
      torch.log(relative_position.float() / max_exact)
      / math.log(max_distance / max_exact)
      * (num_buckets - max_exact)
  ).to(torch.long)
  relative_position_if_large = torch.min(
      relative_position_if_large,
      torch.full_like(relative_position_if_large, num_buckets - 1),
  )

  relative_buckets += torch.where(
      is_small, relative_position, relative_position_if_large
  )
  return relative_buckets


def build_relative_position_buckets(
    query_length: int,
    key_length: int,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
  """Relative position buckets for computing bias.

  Args:
    query_length: an integer of length of current query tensor.
    key_length: an integer of length of current key tensor.
    bidirectional: a boolean - whether the attention is bidirectional, default
      is True.
    num_buckets: an integer for number of buckets, default is 32.
    max_distance: an integer for max distance, default is 128.

  Returns:
    A torch.Tensor of computed relative position buckets.
  """
  context_position = torch.arange(query_length, dtype=torch.long)[:, None]
  memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
  relative_position = (
      memory_position - context_position
  )  # shape (query_length, key_length)
  rel_pos_bucket = relative_position_bucket(
      relative_position,  # shape (query_length, key_length)
      bidirectional=bidirectional,
      num_buckets=num_buckets,
      max_distance=max_distance,
  )
  return rel_pos_bucket.unsqueeze(0).unsqueeze(0)
