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
# Common building blocks for FeedForward layers.

from typing import Callable, Optional

import torch
from torch import nn


class SequentialFeedForward(nn.Module):
  """Vanilla sequential Feedforward with customizable activation."""

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      activation: Callable[[torch.Tensor], torch.Tensor],
      use_bias=False,
      use_glu=False,
      pre_ff_norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
      post_ff_norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
  ):
    """Init function for feedforward layer.

    Args:
      dim (int): embedding size.
      hidden_dim (int): hidden dim size of the feedforward layer.
      activation (Callable): activation function used in this block.
      use_bias (Boolean): whether to use bias. Default is false.
      use_glu (Boolean): whether to use glu in activation. Default is false.
      pre_ff_norm (Callable): pre feedforward norm. Default is None.
      post_ff_norm (Callable): post feedforward norm. Default is None.
    """
    super().__init__()
    self.act = activation
    if use_glu:
      self.w1 = nn.Linear(dim, hidden_dim * 2, bias=use_bias)
    else:
      self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
    self.w2 = nn.Linear(hidden_dim, dim, bias=use_bias)
    self.pre_ff_norm = pre_ff_norm if pre_ff_norm else lambda x: x
    self.post_ff_norm = post_ff_norm if post_ff_norm else lambda x: x

  def forward(self, x):
    """Forward pass for Feedforward layer.

    Args:
      x (torch.Tensor): the input tensor.

    Returns:
      torch.Tensor: output tensor after feedforward.
    """
    x_norm = self.pre_ff_norm(x)
    out = self.w2(self.act(self.w1(x_norm)))
    return self.post_ff_norm(out)


class GatedFeedForward(nn.Module):
  """Gated Feedforward with customizable activation.

  https://arxiv.org/pdf/2002.05202v1.pdf
  """

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      activation: Callable[[torch.Tensor], torch.Tensor],
      use_bias=False,
      use_glu=False,
      pre_ff_norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
      post_ff_norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
  ):
    """Init function for feedforward layer.

    Args:
      dim (int): embedding size.
      hidden_dim (int): hidden dim size of the feedforward layer.
      activation (Callable): activation function used in this block.
      use_bias (Boolean): whether to use bias. Default is false.
      use_glu (Boolean): whether to use glu in activation. Default is false.
      pre_ff_norm (Callable): pre feedforward norm. Default is None.
      post_ff_norm (Callable): post feedforward norm. Default is None.
    """
    super().__init__()
    self.act = activation
    if use_glu:
      self.w1 = nn.Linear(dim, hidden_dim * 2, bias=use_bias)
    else:
      self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
    self.w2 = nn.Linear(hidden_dim, dim, bias=use_bias)
    self.w3 = nn.Linear(dim, hidden_dim, bias=use_bias)
    self.pre_ff_norm = pre_ff_norm if pre_ff_norm else lambda x: x
    self.post_ff_norm = post_ff_norm if post_ff_norm else lambda x: x

  def forward(self, x):
    """Forward pass for Feedforward layer.

    Args:
      x (torch.Tensor): the input tensor.

    Returns:
      torch.Tensor: output tensor after feedforward.
    """
    x_norm = self.pre_ff_norm(x)
    out = self.w2(self.act(self.w1(x_norm)) * self.w3(x_norm))
    return self.post_ff_norm(out)
