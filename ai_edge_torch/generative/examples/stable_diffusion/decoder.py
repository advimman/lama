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

import ai_edge_torch.generative.layers.builder as layers_builder
import ai_edge_torch.generative.layers.model_config as layers_cfg
from ai_edge_torch.generative.layers.unet import blocks_2d
import ai_edge_torch.generative.layers.unet.model_config as unet_cfg
from ai_edge_torch.generative.utilities import stable_diffusion_loader
import torch
from torch import nn

TENSOR_NAMES = stable_diffusion_loader.AutoEncoderModelLoader.TensorNames(
    post_quant_conv="first_stage_model.post_quant_conv",
    conv_in="first_stage_model.decoder.conv_in",
    mid_block_tensor_names=stable_diffusion_loader.MidBlockTensorNames(
        residual_block_tensor_names=[
            stable_diffusion_loader.ResidualBlockTensorNames(
                norm_1="first_stage_model.decoder.mid.block_1.norm1",
                norm_2="first_stage_model.decoder.mid.block_1.norm2",
                conv_1="first_stage_model.decoder.mid.block_1.conv1",
                conv_2="first_stage_model.decoder.mid.block_1.conv2",
            ),
            stable_diffusion_loader.ResidualBlockTensorNames(
                norm_1="first_stage_model.decoder.mid.block_2.norm1",
                norm_2="first_stage_model.decoder.mid.block_2.norm2",
                conv_1="first_stage_model.decoder.mid.block_2.conv1",
                conv_2="first_stage_model.decoder.mid.block_2.conv2",
            ),
        ],
        attention_block_tensor_names=[
            stable_diffusion_loader.AttentionBlockTensorNames(
                norm="first_stage_model.decoder.mid.attn_1.norm",
                q_proj="first_stage_model.decoder.mid.attn_1.q",
                k_proj="first_stage_model.decoder.mid.attn_1.k",
                v_proj="first_stage_model.decoder.mid.attn_1.v",
                output_proj="first_stage_model.decoder.mid.attn_1.proj_out",
            )
        ],
    ),
    up_decoder_blocks_tensor_names=[
        stable_diffusion_loader.UpDecoderBlockTensorNames(
            residual_block_tensor_names=[
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.3.block.0.norm1",
                    norm_2="first_stage_model.decoder.up.3.block.0.norm2",
                    conv_1="first_stage_model.decoder.up.3.block.0.conv1",
                    conv_2="first_stage_model.decoder.up.3.block.0.conv2",
                ),
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.3.block.1.norm1",
                    norm_2="first_stage_model.decoder.up.3.block.1.norm2",
                    conv_1="first_stage_model.decoder.up.3.block.1.conv1",
                    conv_2="first_stage_model.decoder.up.3.block.1.conv2",
                ),
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.3.block.2.norm1",
                    norm_2="first_stage_model.decoder.up.3.block.2.norm2",
                    conv_1="first_stage_model.decoder.up.3.block.2.conv1",
                    conv_2="first_stage_model.decoder.up.3.block.2.conv2",
                ),
            ],
            upsample_conv="first_stage_model.decoder.up.3.upsample.conv",
        ),
        stable_diffusion_loader.UpDecoderBlockTensorNames(
            residual_block_tensor_names=[
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.2.block.0.norm1",
                    norm_2="first_stage_model.decoder.up.2.block.0.norm2",
                    conv_1="first_stage_model.decoder.up.2.block.0.conv1",
                    conv_2="first_stage_model.decoder.up.2.block.0.conv2",
                ),
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.2.block.1.norm1",
                    norm_2="first_stage_model.decoder.up.2.block.1.norm2",
                    conv_1="first_stage_model.decoder.up.2.block.1.conv1",
                    conv_2="first_stage_model.decoder.up.2.block.1.conv2",
                ),
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.2.block.2.norm1",
                    norm_2="first_stage_model.decoder.up.2.block.2.norm2",
                    conv_1="first_stage_model.decoder.up.2.block.2.conv1",
                    conv_2="first_stage_model.decoder.up.2.block.2.conv2",
                ),
            ],
            upsample_conv="first_stage_model.decoder.up.2.upsample.conv",
        ),
        stable_diffusion_loader.UpDecoderBlockTensorNames(
            residual_block_tensor_names=[
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.1.block.0.norm1",
                    norm_2="first_stage_model.decoder.up.1.block.0.norm2",
                    conv_1="first_stage_model.decoder.up.1.block.0.conv1",
                    conv_2="first_stage_model.decoder.up.1.block.0.conv2",
                    residual_layer=(
                        "first_stage_model.decoder.up.1.block.0.nin_shortcut"
                    ),
                ),
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.1.block.1.norm1",
                    norm_2="first_stage_model.decoder.up.1.block.1.norm2",
                    conv_1="first_stage_model.decoder.up.1.block.1.conv1",
                    conv_2="first_stage_model.decoder.up.1.block.1.conv2",
                ),
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.1.block.2.norm1",
                    norm_2="first_stage_model.decoder.up.1.block.2.norm2",
                    conv_1="first_stage_model.decoder.up.1.block.2.conv1",
                    conv_2="first_stage_model.decoder.up.1.block.2.conv2",
                ),
            ],
            upsample_conv="first_stage_model.decoder.up.1.upsample.conv",
        ),
        stable_diffusion_loader.UpDecoderBlockTensorNames(
            residual_block_tensor_names=[
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.0.block.0.norm1",
                    norm_2="first_stage_model.decoder.up.0.block.0.norm2",
                    conv_1="first_stage_model.decoder.up.0.block.0.conv1",
                    conv_2="first_stage_model.decoder.up.0.block.0.conv2",
                    residual_layer=(
                        "first_stage_model.decoder.up.0.block.0.nin_shortcut"
                    ),
                ),
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.0.block.1.norm1",
                    norm_2="first_stage_model.decoder.up.0.block.1.norm2",
                    conv_1="first_stage_model.decoder.up.0.block.1.conv1",
                    conv_2="first_stage_model.decoder.up.0.block.1.conv2",
                ),
                stable_diffusion_loader.ResidualBlockTensorNames(
                    norm_1="first_stage_model.decoder.up.0.block.2.norm1",
                    norm_2="first_stage_model.decoder.up.0.block.2.norm2",
                    conv_1="first_stage_model.decoder.up.0.block.2.conv1",
                    conv_2="first_stage_model.decoder.up.0.block.2.conv2",
                ),
            ],
        ),
    ],
    final_norm="first_stage_model.decoder.norm_out",
    conv_out="first_stage_model.decoder.conv_out",
)


class Decoder(nn.Module):
  """The Decoder model used in Stable Diffusion.

  For details, see https://arxiv.org/abs/2103.00020

  Sturcture of the Decoder:

      latents tensor
            |
            ▼
  ┌───────────────────┐
  │  Post Quant Conv  │
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │      ConvIn       │
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │     MidBlock2D    │
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │    UpDecoder2D    │ x 4
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │     FinalNorm     │
  └─────────┬─────────┘
            |
  ┌─────────▼─────────┐
  │    Activation     │
  └─────────┬─────────┘
            |
  ┌─────────▼─────────┐
  │      ConvOut      │
  └─────────┬─────────┘
            |
            ▼
      Output Image
  """

  def __init__(self, config: unet_cfg.AutoEncoderConfig):
    super().__init__()
    self.config = config
    self.post_quant_conv = nn.Conv2d(
        config.latent_channels,
        config.latent_channels,
        kernel_size=1,
        stride=1,
        padding=0,
    )
    reversed_block_out_channels = list(reversed(config.block_out_channels))
    self.conv_in = nn.Conv2d(
        config.latent_channels,
        reversed_block_out_channels[0],
        kernel_size=3,
        stride=1,
        padding=1,
    )
    self.mid_block = blocks_2d.MidBlock2D(config.mid_block_config)
    up_decoder_blocks = []
    block_out_channels = reversed_block_out_channels[0]
    for i, out_channels in enumerate(reversed_block_out_channels):
      prev_output_channel = block_out_channels
      block_out_channels = out_channels
      not_final_block = i < len(reversed_block_out_channels) - 1
      up_decoder_blocks.append(
          blocks_2d.UpDecoderBlock2D(
              unet_cfg.UpDecoderBlock2DConfig(
                  in_channels=prev_output_channel,
                  out_channels=block_out_channels,
                  normalization_config=config.normalization_config,
                  activation_config=config.activation_config,
                  num_layers=config.layers_per_block,
                  add_upsample=not_final_block,
                  upsample_conv=True,
                  sampling_config=unet_cfg.UpSamplingConfig(
                      mode=unet_cfg.SamplingType.NEAREST, scale_factor=2
                  ),
              )
          )
      )
    self.up_decoder_blocks = nn.ModuleList(up_decoder_blocks)
    self.final_norm = layers_builder.build_norm(
        block_out_channels, config.normalization_config
    )
    self.act_fn = layers_builder.get_activation(config.activation_config)
    self.conv_out = nn.Conv2d(
        block_out_channels,
        config.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )

  def forward(self, latents_tensor: torch.Tensor) -> torch.Tensor:
    """Forward function of decoder model.

    Args:
        latents (torch.Tensor): latents space tensor.

    Returns:
        output decoded image tensor from decoder model.
    """
    x = latents_tensor / self.config.scaling_factor
    x = self.post_quant_conv(x)
    x = self.conv_in(x)
    x = self.mid_block(x)
    for up_decoder_block in self.up_decoder_blocks:
      x = up_decoder_block(x)
    x = self.final_norm(x)
    x = self.act_fn(x)
    x = self.conv_out(x)
    return x


def get_model_config(device_type: str = "cpu") -> unet_cfg.AutoEncoderConfig:
  """Get configs for the Decoder of Stable Diffusion v1.5."""
  in_channels = 3
  latent_channels = 4
  out_channels = 3
  block_out_channels = [128, 256, 512, 512]
  scaling_factor = 0.18215
  layers_per_block = 3

  # For now, only turns on StableHLO composite ops on GPU backend for better
  # performance. CPU should also switch to it once the support is done.
  enable_hlfb = True if device_type == "gpu" else False

  norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.GROUP_NORM,
      group_num=32,
      enable_hlfb=enable_hlfb,
  )

  att_config = unet_cfg.AttentionBlock2DConfig(
      dim=block_out_channels[-1],
      normalization_config=norm_config,
      attention_config=layers_cfg.AttentionConfig(
          num_heads=1,
          head_dim=block_out_channels[-1],
          num_query_groups=1,
          qkv_use_bias=True,
          output_proj_use_bias=True,
          enable_kv_cache=False,
          qkv_transpose_before_split=True,
          qkv_fused_interleaved=False,
          rotary_base=0,
          rotary_percentage=0.0,
      ),
      enable_hlfb=enable_hlfb,
  )

  mid_block_config = unet_cfg.MidBlock2DConfig(
      in_channels=block_out_channels[-1],
      normalization_config=norm_config,
      activation_config=layers_cfg.ActivationConfig(
          layers_cfg.ActivationType.SILU
      ),
      num_layers=1,
      attention_block_config=att_config,
  )

  config = unet_cfg.AutoEncoderConfig(
      in_channels=in_channels,
      latent_channels=latent_channels,
      out_channels=out_channels,
      activation_config=layers_cfg.ActivationConfig(
          layers_cfg.ActivationType.SILU
      ),
      block_out_channels=block_out_channels,
      scaling_factor=scaling_factor,
      layers_per_block=layers_per_block,
      normalization_config=norm_config,
      mid_block_config=mid_block_config,
  )
  return config


def get_fake_model_config(
    device_type: str = "cpu",
) -> unet_cfg.AutoEncoderConfig:
  """Get fake configs for the Decoder of Stable Diffusion v1.5 for testing."""
  in_channels = 3
  latent_channels = 4
  out_channels = 3
  block_out_channels = [2, 4]
  scaling_factor = 0.18215
  layers_per_block = 2

  # For now, only turns on StableHLO composite ops on GPU backend for better
  # performance. CPU should also switch to it once the support is done.
  enable_hlfb = True if device_type == "gpu" else False

  norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.GROUP_NORM,
      group_num=2,
      enable_hlfb=enable_hlfb,
  )

  att_config = unet_cfg.AttentionBlock2DConfig(
      dim=block_out_channels[-1],
      normalization_config=norm_config,
      attention_config=layers_cfg.AttentionConfig(
          num_heads=1,
          head_dim=block_out_channels[-1],
          num_query_groups=1,
          qkv_use_bias=True,
          output_proj_use_bias=True,
          enable_kv_cache=False,
          qkv_transpose_before_split=True,
          qkv_fused_interleaved=False,
          rotary_base=0,
          rotary_percentage=0.0,
      ),
      enable_hlfb=enable_hlfb,
  )

  mid_block_config = unet_cfg.MidBlock2DConfig(
      in_channels=block_out_channels[-1],
      normalization_config=norm_config,
      activation_config=layers_cfg.ActivationConfig(
          layers_cfg.ActivationType.SILU
      ),
      num_layers=1,
      attention_block_config=att_config,
  )

  config = unet_cfg.AutoEncoderConfig(
      in_channels=in_channels,
      latent_channels=latent_channels,
      out_channels=out_channels,
      activation_config=layers_cfg.ActivationConfig(
          layers_cfg.ActivationType.SILU
      ),
      block_out_channels=block_out_channels,
      scaling_factor=scaling_factor,
      layers_per_block=layers_per_block,
      normalization_config=norm_config,
      mid_block_config=mid_block_config,
  )
  return config
