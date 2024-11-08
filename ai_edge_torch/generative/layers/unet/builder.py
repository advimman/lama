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
# Builder utils for individual components.

import ai_edge_torch.generative.layers.unet.model_config as unet_config
from torch import nn


def build_upsampling(config: unet_config.UpSamplingConfig):
  if config.mode == unet_config.SamplingType.NEAREST:
    return nn.UpsamplingNearest2d(scale_factor=config.scale_factor)
  elif config.mode == unet_config.SamplingType.BILINEAR:
    return nn.UpsamplingBilinear2d(scale_factor=config.scale_factor)
  else:
    raise ValueError("Unsupported upsampling type.")


def build_downsampling(config: unet_config.DownSamplingConfig):
  if config.mode == unet_config.SamplingType.AVERAGE:
    return nn.AvgPool2d(
        config.kernel_size, config.stride, padding=config.padding
    )
  elif config.mode == unet_config.SamplingType.CONVOLUTION:
    out_channels = (
        config.in_channels
        if config.out_channels is None
        else config.out_channels
    )
    padding = (0, 1, 0, 1) if config.padding == 0 else config.padding
    return nn.Conv2d(
        config.in_channels,
        out_channels=out_channels,
        kernel_size=config.kernel_size,
        stride=config.stride,
        padding=padding,
    )
  else:
    raise ValueError("Unsupported downsampling type.")
