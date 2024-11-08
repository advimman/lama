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

# UNet configuration class.
import dataclasses
import enum
from typing import List, Optional

import ai_edge_torch.generative.layers.model_config as layers_cfg


@enum.unique
class SamplingType(enum.Enum):
  NEAREST = enum.auto()
  BILINEAR = enum.auto()
  AVERAGE = enum.auto()
  CONVOLUTION = enum.auto()


@dataclasses.dataclass
class UpSamplingConfig:
  mode: SamplingType
  scale_factor: float


@dataclasses.dataclass
class DownSamplingConfig:
  mode: SamplingType
  in_channels: int
  kernel_size: int
  stride: int
  padding: int
  out_channels: Optional[int] = None


@dataclasses.dataclass
class ResidualBlock2DConfig:
  in_channels: int
  hidden_channels: int
  out_channels: int
  normalization_config: layers_cfg.NormalizationConfig
  activation_config: layers_cfg.ActivationConfig
  # Optional time embedding channels if the residual block takes a time embedding context as input
  time_embedding_channels: Optional[int] = None


@dataclasses.dataclass
class AttentionBlock2DConfig:
  dim: int
  normalization_config: layers_cfg.NormalizationConfig
  attention_config: layers_cfg.AttentionConfig
  enable_hlfb: bool = True
  attention_batch_size: int = 1


@dataclasses.dataclass
class CrossAttentionBlock2DConfig:
  query_dim: int
  cross_dim: int
  hidden_dim: int
  output_dim: int
  normalization_config: layers_cfg.NormalizationConfig
  attention_config: layers_cfg.AttentionConfig
  enable_hlfb: bool = True
  attention_batch_size: int = 1


@dataclasses.dataclass
class FeedForwardBlock2DConfig:
  dim: int
  hidden_dim: int
  normalization_config: layers_cfg.NormalizationConfig
  activation_config: layers_cfg.ActivationConfig
  use_bias: bool


@dataclasses.dataclass
class TransformerBlock2DConfig:
  pre_conv_normalization_config: layers_cfg.NormalizationConfig
  attention_block_config: AttentionBlock2DConfig
  cross_attention_block_config: CrossAttentionBlock2DConfig
  feed_forward_block_config: FeedForwardBlock2DConfig


@dataclasses.dataclass
class UpDecoderBlock2DConfig:
  in_channels: int
  out_channels: int
  normalization_config: layers_cfg.NormalizationConfig
  activation_config: layers_cfg.ActivationConfig
  num_layers: int
  # Optional time embedding channels if the residual blocks take a time embedding as input
  time_embedding_channels: Optional[int] = None
  # Whether to add upsample operation after residual blocks
  add_upsample: bool = True
  # Whether to add a conv2d layer after upsample
  upsample_conv: bool = True
  # Optional sampling config if add_upsample is True.
  sampling_config: Optional[UpSamplingConfig] = None
  # Optional config of transformer blocks interleaved with residual blocks
  transformer_block_config: Optional[TransformerBlock2DConfig] = None
  # Optional dimension of context tensor if context tensor is given as input.
  context_dim: Optional[int] = None


@dataclasses.dataclass
class SkipUpDecoderBlock2DConfig:
  in_channels: int
  out_channels: int
  # The dimension of output channels of previous connected block
  prev_out_channels: int
  normalization_config: layers_cfg.NormalizationConfig
  activation_config: layers_cfg.ActivationConfig
  num_layers: int
  # Optional time embedding channels if the residual blocks take a time embedding as input
  time_embedding_channels: Optional[int] = None
  # Whether to add upsample operation after residual blocks
  add_upsample: bool = True
  # Whether to add a conv2d layer after upsample
  upsample_conv: bool = True
  # Optional sampling config if add_upsample is True.
  sampling_config: Optional[UpSamplingConfig] = None
  # Optional config of transformer blocks interleaved with residual blocks
  transformer_block_config: Optional[TransformerBlock2DConfig] = None
  # Optional dimension of context tensor if context tensor is given as input.
  context_dim: Optional[int] = None


@dataclasses.dataclass
class DownEncoderBlock2DConfig:
  in_channels: int
  out_channels: int
  normalization_config: layers_cfg.NormalizationConfig
  activation_config: layers_cfg.ActivationConfig
  num_layers: int
  # Padding for the downsampling convolution.
  padding: int = 1
  # Optional time embedding channels if the residual blocks take a time embedding as input
  time_embedding_channels: Optional[int] = None
  # Whether to add downsample operation after residual blocks
  add_downsample: bool = True
  # Optional sampling config if add_upsample is True.
  sampling_config: Optional[DownSamplingConfig] = None
  # Optional config of transformer blocks interleaved with residual blocks
  transformer_block_config: Optional[TransformerBlock2DConfig] = None
  # Optional dimension of context tensor if context tensor is given as input.
  context_dim: Optional[int] = None


@dataclasses.dataclass
class MidBlock2DConfig:
  in_channels: int
  normalization_config: layers_cfg.NormalizationConfig
  activation_config: layers_cfg.ActivationConfig
  num_layers: int
  # Optional time embedding channels if the residual blocks take a time embedding context as input
  time_embedding_channels: Optional[int] = None
  # Optional config of attention blocks interleaved with residual blocks
  attention_block_config: Optional[AttentionBlock2DConfig] = None
  # Optional config of transformer blocks interleaved with residual blocks
  transformer_block_config: Optional[TransformerBlock2DConfig] = None
  # Optional dimension of context tensor if context tensor is given as input.
  context_dim: Optional[int] = None


@dataclasses.dataclass
class AutoEncoderConfig:
  """Configurations of encoder/decoder in the autoencoder model."""

  # The activation type of encoder/decoder blocks.
  activation_config: layers_cfg.ActivationConfig

  # The output channels of each block.
  block_out_channels: List[int]

  # Number of channels in the input image.
  in_channels: int

  # Number of channels in the output.
  out_channels: int

  # Number of channels in the latent space.
  latent_channels: int

  # The component-wise standard deviation of the trained latent space computed using the first batch of the
  # training set. This is used to scale the latent space to have unit variance when training the diffusion
  # model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
  # diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
  # / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
  # Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
  scaling_factor: float

  # The layesr number of each encoder/decoder block.
  layers_per_block: int

  # The normalization config.
  normalization_config: layers_cfg.NormalizationConfig

  # The configuration of middle blocks, that is, after the last block of encoder and before the first block of decoder.
  mid_block_config: MidBlock2DConfig


@dataclasses.dataclass
class DiffusionModelConfig:
  """Configurations of Diffusion model."""

  # Number of channels in the input tensor.
  in_channels: int

  # Number of channels in the output tensor.
  out_channels: int

  # The output channels of each block.
  block_out_channels: List[int]

  # The layesr number of each block.
  layers_per_block: int

  # The padding to use for the downsampling.
  downsample_padding: int

  # Normalization config used in residual blocks.
  residual_norm_config: layers_cfg.NormalizationConfig

  # Activation config used in residual blocks
  residual_activation_type: layers_cfg.ActivationType

  # The batch size used in transformer blocks, for attention layers.
  transformer_batch_size: int

  # The number of attention heads used in transformer blocks.
  transformer_num_attention_heads: int

  # The dimension of cross attention used in transformer blocks.
  transformer_cross_attention_dim: int

  # Normalization config used in prev conv layer of transformer blocks.
  transformer_pre_conv_norm_config: layers_cfg.NormalizationConfig

  # Normalization config used in transformer blocks.
  transformer_norm_config: layers_cfg.NormalizationConfig

  # Activation type of feed forward used in transformer blocks.
  transformer_ff_activation_type: layers_cfg.ActivationType

  # Number of layers in mid block.
  mid_block_layers: int

  # Dimension of time embedding.
  time_embedding_dim: int

  # Time embedding dimensions for blocks.
  time_embedding_blocks_dim: int

  # Normalization config used for final layer
  final_norm_config: layers_cfg.NormalizationConfig

  # Activation type used in final layer
  final_activation_type: layers_cfg.ActivationType

  # Whether to enable StableHLO composite ops in the model.
  enable_hlfb: bool = False
