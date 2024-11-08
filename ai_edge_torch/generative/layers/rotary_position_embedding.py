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
# Implementation for Rotary Position embedding. https://arxiv.org/pdf/2104.09864.pdf
import torch


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
  """Computes rotary positional embedding.

  Args:
    x: the input tensor.
    cos: cosine value for the rope.
    sin: sin value for the rope.

  Returns:
    output tensor of RoPE.
  """
  x = x.transpose(1, 2)
  head_size = x.size(-1)
  x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
  x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
  rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
  roped = (x * cos) + (rotated * sin)
  return roped.transpose(1, 2).type_as(x)
