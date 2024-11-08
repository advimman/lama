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

from ai_edge_torch.generative.examples.stable_diffusion.attention import SelfAttention  # NOQA
import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):

  def __init__(self, channels):
    super().__init__()
    self.groupnorm = nn.GroupNorm(32, channels)
    self.attention = SelfAttention(1, channels)

  def forward(self, x):
    residue = x
    x = self.groupnorm(x)

    n, c, h, w = x.shape
    x = x.view((n, c, h * w))
    x = x.transpose(-1, -2)
    x = self.attention(x)
    x = x.transpose(-1, -2)
    x = x.view((n, c, h, w))

    x += residue
    return x


class ResidualBlock(nn.Module):

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.groupnorm_1 = nn.GroupNorm(32, in_channels)
    self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    self.groupnorm_2 = nn.GroupNorm(32, out_channels)
    self.conv_2 = nn.Conv2d(
        out_channels, out_channels, kernel_size=3, padding=1
    )

    if in_channels == out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(
          in_channels, out_channels, kernel_size=1, padding=0
      )

  def forward(self, x):
    residue = x

    x = self.groupnorm_1(x)
    x = F.silu(x)
    x = self.conv_1(x)

    x = self.groupnorm_2(x)
    x = F.silu(x)
    x = self.conv_2(x)

    return x + self.residual_layer(residue)


class Encoder(nn.Sequential):

  def __init__(self):
    super().__init__(
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        ResidualBlock(128, 128),
        ResidualBlock(128, 128),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        ResidualBlock(128, 256),
        ResidualBlock(256, 256),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
        ResidualBlock(256, 512),
        ResidualBlock(512, 512),
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        AttentionBlock(512),
        ResidualBlock(512, 512),
        nn.GroupNorm(32, 512),
        nn.SiLU(),
        nn.Conv2d(512, 8, kernel_size=3, padding=1),
        nn.Conv2d(8, 8, kernel_size=1, padding=0),
    )

  @torch.inference_mode
  def forward(self, x, noise):
    for module in self:
      if getattr(module, 'stride', None) == (
          2,
          2,
      ):  # Padding at downsampling should be asymmetric (see #8)
        x = F.pad(x, (0, 1, 0, 1))
      x = module(x)

    mean, log_variance = torch.chunk(x, 2, dim=1)
    log_variance = torch.clamp(log_variance, -30, 20)
    variance = log_variance.exp()
    stdev = variance.sqrt()
    x = mean + stdev * noise

    x *= 0.18215
    return x
