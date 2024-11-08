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
"""Transforms the input and output of a module to channel last layout."""

from typing import Optional

import torch
from torch import nn


class ChannelLastIOWrapper(nn.Module):

  def __init__(self, wrapped, *, args=None, outputs=None):
    super().__init__()
    self.wrapped = wrapped
    self._args = args or []
    self._outputs = outputs or []

  def _to_channel_last(self, x):
    if not torch.is_tensor(x):
      raise ValueError("Input must be a torch tensor")
    if x.ndim < 3:
      raise ValueError(
          "Input must be a tensor with rank >= 3 in layout (N, C, ...)"
      )
    dims = [0, *range(2, x.ndim), 1]
    return torch.permute(x, dims)

  def _to_channel_first(self, x):
    if not torch.is_tensor(x):
      raise ValueError("Input must be a torch tensor.")
    if x.ndim < 3:
      raise ValueError(
          "Input must be a tensor with rank >= 3 in layout (N, ..., C)"
      )
    dims = [0, x.ndim - 1, *range(1, x.ndim - 1)]
    return torch.permute(x, dims)

  def forward(self, *args, **kwargs):
    args = list(args)
    for i in self._args:
      args[i] = self._to_channel_first(args[i])

    outputs = self.wrapped(*args, **kwargs)

    if not isinstance(outputs, (list, tuple)):
      outputs_is_list = False
      output_list = [outputs]
    else:
      outputs_is_list = True
      output_list = list(outputs)

    for i in self._outputs:
      output_list[i] = self._to_channel_last(output_list[i])

    if not outputs_is_list:
      return output_list[0]
    else:
      return type(outputs)(output_list)


def to_channel_last_io(
    module: nn.Module,
    args: Optional[list[int]] = None,
    outputs: Optional[list[int]] = None,
):
  """Wraps the module with channel first to channel last layout transformations.

  Args:
    args (list[int]): Transform args with indices in the list from channel first
      (N, C, ...) to channel last (N, ..., C).
    outputs (list[int]): Transform outputs with indices in the list from channel
      first (N, C, ...) to channel last (N, ..., C).

  Returns:
    The wrapped nn.Module with additional layout transposes after inputs and/or
    before
    outputs.
  """
  return ChannelLastIOWrapper(module, args=args, outputs=outputs)
