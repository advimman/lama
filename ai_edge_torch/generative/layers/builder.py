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
# Builder class for individual components.
from typing import Callable

import ai_edge_torch.generative.layers.feed_forward as feed_forward
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.normalization as normalization
import torch
from torch import nn
import torch.nn.functional as F


class GeGLU(nn.Module):
  """GeGLU is an activation function which is a variant of GELU.

  GeGLU(x) = (xW+b) * GELU(xV+c)
  See: https://arxiv.org/abs/2002.05202v1
  """

  def __init__(self, d_in: int, d_out: int):
    super().__init__()
    self.proj = nn.Linear(d_in, d_out * 2)

  def forward(self, x: torch.Tensor):
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * F.gelu(gate)


class SwiGLU(nn.Module):
  """SwiGLU is an activation function which is a variant of GLU.

  SwiGLU is same as SiLU_GLU, because The SiLU function is also known as the
  swish function.

  SwiGLU(x) = Swish(xW+b) * (xV+c)
  See: https://paperswithcode.com/method/swiglu
  """

  def forward(self, x: torch.Tensor):
    x, y = x.chunk(2, dim=-1)
    return F.silu(x) * y


def build_norm(dim: int, config: cfg.NormalizationConfig):
  """Builder function for normalizers.

  Args:
    dim (int): dimension of the input tensor.
    config (`NormalizationConfig` object): the normalization configuration.

  Returns:
    The constructed `nn.Module` normalization layer.

  Raises:
    ValueError: If config's `layer_norm_type` is not supported.
  """
  if config.type == cfg.NormalizationType.NONE:
    return lambda x: x
  elif config.type == cfg.NormalizationType.RMS_NORM:
    return normalization.RMSNorm(
        dim,
        eps=config.epsilon,
        zero_centered_gamma=config.zero_centered,
    )
  elif config.type == cfg.NormalizationType.LAYER_NORM:
    return normalization.LayerNorm(dim, config.epsilon, config.enable_hlfb)
  elif config.type == cfg.NormalizationType.GROUP_NORM:
    return normalization.GroupNorm(
        config.group_num, dim, config.epsilon, config.enable_hlfb
    )
  else:
    raise ValueError("Unsupported norm type.")


def build_ff(dim: int, config: cfg.FeedForwardConfig):
  """Builder function for Feed Forward. Supports `Sequential` and `Gated`.

  Args:
    dim (int): dimension of the input tensor.
    config (`FeedForwardConfig` object): the model configuration.

  Returns:
    The constructed `nn.Module` feedforward layer.

  Raises:
    ValueError: If config's `ff_type` is not supported.
  """
  ff_type = config.type
  if ff_type == cfg.FeedForwardType.SEQUENTIAL:
    ff_module = feed_forward.SequentialFeedForward
  elif ff_type == cfg.FeedForwardType.GATED:
    ff_module = feed_forward.GatedFeedForward
  else:
    raise ValueError("Unsupported feedforward type.")

  activation = get_activation(config.activation)

  pre_ff_norm = build_norm(dim, config.pre_ff_norm_config)
  post_ff_norm = build_norm(dim, config.post_ff_norm_config)

  return ff_module(
      dim=dim,
      hidden_dim=config.intermediate_size,
      activation=activation,
      use_bias=config.use_bias,
      use_glu=(
          config.activation.type == cfg.ActivationType.GE_GLU
          or config.activation.type == cfg.ActivationType.SILU_GLU
      ),
      pre_ff_norm=pre_ff_norm,
      post_ff_norm=post_ff_norm,
  )


def get_activation(config: cfg.ActivationConfig):
  """Get pytorch callable activation from the activation config.

  Args:
    config (cfg.ActivationConfig): activation config.

  Returns:
    Activation function.

  Raises:
    ValueError: If activation config is not supported.
  """
  if config.type == cfg.ActivationType.LINEAR:
    return lambda x: x
  elif config.type == cfg.ActivationType.SILU:
    return F.silu
  elif config.type == cfg.ActivationType.GELU:
    return F.gelu
  elif config.type == cfg.ActivationType.GELU_TANH:
    return lambda x: F.gelu(x, approximate="tanh")
  elif config.type == cfg.ActivationType.GELU_QUICK:
    # GELU approximation that is fast but somewhat inaccurate.
    # See: https://github.com/hendrycks/GELUs
    return lambda x: x * F.sigmoid(1.702 * x)
  elif config.type == cfg.ActivationType.GE_GLU:
    return GeGLU(config.dim_in, config.dim_out)
  elif config.type == cfg.ActivationType.RELU:
    return F.relu
  elif config.type == cfg.ActivationType.SILU_GLU:
    return SwiGLU()
  else:
    raise ValueError("Unsupported activation type.")
