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

import math

import torch
from torch import _decomp
from torch import nn
from torch._prims_common import mask_tensor
from torch._prims_common.wrappers import out_wrapper
from torch.nn import functional as F


def triu(a):
  h, w = a.shape[-2:]
  mask = (
      torch.arange(w, device=a.device).unsqueeze(-2)
      - torch.arange(h, device=a.device).unsqueeze(-1)
  ) >= 1
  mask = torch.broadcast_to(mask, a.shape)
  return torch.ops.aten.logical_and(a, mask).contiguous()


# _decomp.decomposition_table[torch.ops.aten.triu.default] = triu


class SelfAttention(nn.Module):

  def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
    super().__init__()
    self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
    self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    self.n_heads = n_heads
    self.d_head = d_embed // n_heads

  def forward(self, x, causal_mask=False):
    input_shape = x.shape
    batch_size, sequence_length, d_embed = input_shape
    interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

    q, k, v = self.in_proj(x).chunk(3, dim=-1)

    q = q.view(interim_shape).transpose(1, 2)
    k = k.view(interim_shape).transpose(1, 2)
    v = v.view(interim_shape).transpose(1, 2)

    weight = q @ k.transpose(-1, -2)
    if causal_mask:
      # mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
      mask = triu(torch.ones_like(weight, dtype=torch.bool))
      weight.masked_fill_(mask, -torch.inf)
    weight /= math.sqrt(self.d_head)
    weight = F.softmax(weight, dim=-1)

    output = weight @ v
    output = output.transpose(1, 2)
    output = output.reshape(input_shape)
    output = self.out_proj(output)
    return output


class CrossAttention(nn.Module):

  def __init__(
      self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True
  ):
    super().__init__()
    self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
    self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
    self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
    self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    self.n_heads = n_heads
    self.d_head = d_embed // n_heads

  def forward(self, x, y):
    input_shape = x.shape
    batch_size, sequence_length, d_embed = input_shape
    interim_shape = (batch_size, -1, self.n_heads, self.d_head)

    q = self.q_proj(x)
    k = self.k_proj(y)
    v = self.v_proj(y)

    q = q.view(interim_shape).transpose(1, 2)
    k = k.view(interim_shape).transpose(1, 2)
    v = v.view(interim_shape).transpose(1, 2)

    weight = q @ k.transpose(-1, -2)
    weight /= math.sqrt(self.d_head)
    weight = F.softmax(weight, dim=-1)

    output = weight @ v
    output = output.transpose(1, 2).contiguous()
    output = output.view(input_shape)
    output = self.out_proj(output)
    return output
