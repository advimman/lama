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
# Common normalization layers.

from ai_edge_torch.hlfb import StableHLOCompositeBuilder
import torch
from torch import nn
import torch.nn.functional as F


# Implementation for RMSNorm from: https://arxiv.org/abs/1910.07467
class RMSNorm(torch.nn.Module):

  def __init__(self, dim: int, eps: float = 1e-6, zero_centered_gamma=False):
    """Initialize the RMSNorm layer.

    Args:
      dim (int): dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default:
        1e-6).
    """
    super().__init__()
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.ones(dim))
    self.zero_centered_gamma = zero_centered_gamma

  def _norm(self, x):
    """Apply RMSNorm normalization.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: The normalized output tensor.
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    """Running the forward pass of RMSNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying RMSNorm.
    """
    output = self._norm(x.float()).type_as(x)
    if self.zero_centered_gamma:
      return output * (1 + self.weight)
    else:
      return output * self.weight


class GroupNorm(torch.nn.Module):

  def __init__(
      self,
      group_num: int,
      dim: int,
      eps: float = 1e-5,
      enable_hlfb: bool = False,
  ):
    """Initialize the GroupNorm layer.

    Args:
      group_num (int): Number of groups to separate the channels into.
      dim (int): Dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default:
        1e-5).
      enable_hlfb (bool): Whether to convert this normalization into a single
        op.
    """
    super().__init__()
    self.enable_hlfb = enable_hlfb
    self.group_num = group_num
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.empty(dim))
    self.bias = torch.nn.Parameter(torch.empty(dim))

  def forward(self, x):
    """Running the forward pass of GroupNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying GroupNorm.
    """
    if self.enable_hlfb:
      return group_norm_with_hlfb(
          x,
          self.weight,
          self.bias,
          self.group_num,
          self.eps,
      )
    else:
      return F.group_norm(x, self.group_num, self.weight, self.bias, self.eps)


class LayerNorm(torch.nn.Module):

  def __init__(
      self,
      dim: int,
      eps: float = 1e-5,
      enable_hlfb: bool = False,
  ):
    """Initialize the LayerNorm layer.

    Args:
      dim (int): dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default:
        1e-5).
      enable_hlfb (bool): Whether to convert this normalization into a single
        op.
    """
    super().__init__()
    self.enable_hlfb = enable_hlfb
    self.normalized_shape = (dim,)
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.empty(dim))
    self.bias = torch.nn.Parameter(torch.empty(dim))

  def forward(self, x):
    """Running the forward pass of LayerNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying LayerNorm.
    """
    if self.enable_hlfb:
      return layer_norm_with_hlfb(
          x, self.normalized_shape, self.weight, self.bias, self.eps
      )
    return F.layer_norm(
        x, self.normalized_shape, self.weight, self.bias, self.eps
    )


def group_norm_with_hlfb(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    num_groups: int,
    eps: float,
):
  """Group Normalization with high-level function boundary enabled.

  Args:
    x (torch.Tensor): Input tensor for Group Normalization, with BCHW shape.
    w (torch.Tensor): The weight tensor for the normalization.
    b (torch.Tensor): The bias tensor for the normalization.
    num_groups (int): Number of groups to separate the channels into.
    eps (float): A small float value to ensure numerical stability.

  Returns:
    The output tensor of Group Normalization.
  """
  x = torch.permute(x, (0, 2, 3, 1))

  # TODO: b/366544750 - Change "reduction_axes" field as an array, rather than
  # int32 when the bug is fixed.
  builder = StableHLOCompositeBuilder(
      name="odml.group_norm",
      attr={
          "num_groups": num_groups,
          "epsilon": eps,
          "reduction_axes": 3,
          "channel_axis": 3,
      },
  )
  x, w, b = builder.mark_inputs(x, w, b)
  x = torch.permute(x, (0, 3, 1, 2))
  y = F.group_norm(x, num_groups, weight=w, bias=b, eps=eps)
  y = torch.permute(y, (0, 2, 3, 1))
  y = builder.mark_outputs(y)

  y = torch.permute(y, (0, 3, 1, 2))
  return y


def layer_norm_with_hlfb(
    x: torch.Tensor,
    normalized_shape: list[int],
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float,
):
  """Layer Normalization with high-level function boundary enabled.

  Args:
    x (torch.Tensor): Input tensor for Layer Normalization, with BCHW shape.
    normalized_shape (list[int]): Input shape from an expected input of size,
      same as https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html.
    w (torch.Tensor): The weight tensor for the normalization.
    b (torch.Tensor): The bias tensor for the normalization.
    eps (float): A small float value to ensure numerical stability.

  Returns:
    The output tensor of Layer Normalization.
  """
  builder = StableHLOCompositeBuilder(
      name="odml.group_norm",
      attr={"num_groups": 1, "epsilon": eps, "channel_axis": 1},
  )
  x, w, b = builder.mark_inputs(x, w, b)
  y = F.layer_norm(x, normalized_shape, w, b, eps=eps)
  y = builder.mark_outputs(y)
  return y
