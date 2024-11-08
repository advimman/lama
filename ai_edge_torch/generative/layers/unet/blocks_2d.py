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

from typing import List, Optional, Tuple, Union

from ai_edge_torch.generative.layers.attention import CrossAttention
from ai_edge_torch.generative.layers.attention import SelfAttention
import ai_edge_torch.generative.layers.builder as layers_builder
import ai_edge_torch.generative.layers.model_config as layers_cfg
import ai_edge_torch.generative.layers.unet.builder as unet_builder
import ai_edge_torch.generative.layers.unet.model_config as unet_cfg
import torch
from torch import nn


class ResidualBlock2D(nn.Module):
  """2D Residual block containing two Conv2D with optional time embedding as input."""

  def __init__(self, config: unet_cfg.ResidualBlock2DConfig):
    """Initialize an instance of the ResidualBlock2D.

    Args:
      config (unet_cfg.ResidualBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    self.norm_1 = layers_builder.build_norm(
        config.in_channels, config.normalization_config
    )
    self.conv_1 = nn.Conv2d(
        config.in_channels,
        config.hidden_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    if config.time_embedding_channels is not None:
      self.time_emb_proj = nn.Linear(
          config.time_embedding_channels, config.hidden_channels
      )
    else:
      self.time_emb_proj = None
    self.norm_2 = layers_builder.build_norm(
        config.hidden_channels, config.normalization_config
    )
    self.conv_2 = nn.Conv2d(
        config.hidden_channels,
        config.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    self.act_fn = layers_builder.get_activation(config.activation_config)
    if config.in_channels == config.out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(
          config.in_channels,
          config.out_channels,
          kernel_size=1,
          stride=1,
          padding=0,
      )

  def forward(
      self, input_tensor: torch.Tensor, time_emb: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    """Forward function of the ResidualBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (Optional[torch.Tensor]): optional time embedding tensor.

    Returns:
      output hidden_states tensor after ResidualBlock2D.
    """
    residual = input_tensor
    x = self.norm_1(input_tensor)
    x = self.act_fn(x)
    x = self.conv_1(x)
    if self.time_emb_proj is not None:
      time_emb = self.act_fn(time_emb)
      time_emb = self.time_emb_proj(time_emb)[:, :, None, None]
      x = x + time_emb
    x = self.norm_2(x)
    x = self.act_fn(x)
    x = self.conv_2(x)
    x = x + self.residual_layer(residual)
    return x


class AttentionBlock2D(nn.Module):
  """2D self attention block

  x = SelfAttention(Norm(input_tensor)) + x
  """

  def __init__(self, config: unet_cfg.AttentionBlock2DConfig):
    """Initialize an instance of the AttentionBlock2D.

    Args:
      config (unet_cfg.AttentionBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    self.norm = layers_builder.build_norm(
        config.dim, config.normalization_config
    )
    self.attention = SelfAttention(
        config.attention_batch_size,
        config.dim,
        config.attention_config,
        enable_hlfb=config.enable_hlfb,
    )

  def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """Forward function of the AttentionBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.

    Returns:
      output activation tensor after self attention.
    """
    residual = input_tensor
    B, C, H, W = input_tensor.shape
    if (
        self.config.normalization_config.type
        == layers_cfg.NormalizationType.GROUP_NORM
    ):
      x = self.norm(input_tensor)
      x = x.view(B, C, H * W)
      x = x.transpose(-1, -2)
    else:
      x = torch.permute(input_tensor, (0, 2, 3, 1))
      x = self.norm(x)
      x = x.view(B, H * W, C)
    x = x.contiguous()  # Prevent BATCH_MATMUL op in converted tflite.
    x = self.attention(x)
    x = x.view(B, H, W, C)
    residual = torch.permute(residual, (0, 2, 3, 1))
    x = x + residual
    x = torch.permute(x, (0, 3, 1, 2))
    return x


class CrossAttentionBlock2D(nn.Module):
  """2D cross attention block

  x = CrossAttention(Norm(input_tensor), context) + x
  """

  def __init__(self, config: unet_cfg.CrossAttentionBlock2DConfig):
    """Initialize an instance of the AttentionBlock2D.

    Args:
      config (unet_cfg.CrossAttentionBlock2DConfig): the configuration of this
        block.
    """
    super().__init__()
    self.config = config
    self.norm = layers_builder.build_norm(
        config.query_dim, config.normalization_config
    )
    self.attention = CrossAttention(
        config.attention_batch_size,
        config.query_dim,
        config.cross_dim,
        config.hidden_dim,
        config.output_dim,
        config.attention_config,
        enable_hlfb=config.enable_hlfb,
    )

  def forward(
      self, input_tensor: torch.Tensor, context_tensor: torch.Tensor
  ) -> torch.Tensor:
    """Forward function of the CrossAttentionBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      context_tensor (torch.Tensor): the context tensor to apply cross attention
        on.

    Returns:
      output activation tensor after cross attention.
    """
    residual = input_tensor
    B, C, H, W = input_tensor.shape
    if (
        self.config.normalization_config.type
        == layers_cfg.NormalizationType.GROUP_NORM
    ):
      x = self.norm(input_tensor)
      x = x.view(B, C, H * W)
      x = x.transpose(-1, -2)
    else:
      x = torch.permute(input_tensor, (0, 2, 3, 1))
      x = self.norm(x)
      x = x.view(B, H * W, C)
    x = self.attention(x, context_tensor)
    x = x.view(B, H, W, C)
    residual = torch.permute(residual, (0, 2, 3, 1))
    x = x + residual
    x = torch.permute(x, (0, 3, 1, 2))
    return x


class FeedForwardBlock2D(nn.Module):
  """2D feed forward block

  x = w2(Activation(w1(Norm(x)))) + x
  """

  def __init__(
      self,
      config: unet_cfg.FeedForwardBlock2DConfig,
  ):
    super().__init__()
    self.config = config
    self.act = layers_builder.get_activation(config.activation_config)
    self.norm = layers_builder.build_norm(
        config.dim, config.normalization_config
    )
    if config.activation_config.type == layers_cfg.ActivationType.GE_GLU:
      self.w1 = nn.Identity()
      self.w2 = nn.Linear(config.hidden_dim, config.dim)
    else:
      self.w1 = nn.Linear(config.dim, config.hidden_dim)
      self.w2 = nn.Linear(config.hidden_dim, config.dim)

  def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
    residual = input_tensor
    B, C, H, W = input_tensor.shape
    if (
        self.config.normalization_config.type
        == layers_cfg.NormalizationType.GROUP_NORM
    ):
      x = self.norm(input_tensor)
      x = x.view(B, C, H * W)
      x = x.transpose(-1, -2)
    else:
      x = torch.permute(input_tensor, (0, 2, 3, 1))
      x = self.norm(x)
      x = x.view(B, H * W, C)
    x = self.w1(x)
    x = self.act(x)
    x = self.w2(x)
    x = x.view(B, H, W, C)
    residual = torch.permute(residual, (0, 2, 3, 1))
    x = x + residual
    x = torch.permute(x, (0, 3, 1, 2))
    return x


class TransformerBlock2D(nn.Module):
  """Basic transformer block used in UNet of diffusion model

       input_tensor    context_tensor
            |                 |
  ┌─────────▼─────────┐       |
  │      ConvIn       |       │
  └─────────┬─────────┘       |
            |                 |
            ▼                 |
  ┌───────────────────┐       |
  │  Attention Block  │       |
  └─────────┬─────────┘       |
            │                 |
  ┌────────────────────┐      |
  │CrossAttention Block│◄─────┘
  └─────────┬──────────┘
            │
  ┌─────────▼─────────┐
  │ FeedForwardBlock  │
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │      ConvOut      │
  └─────────┬─────────┘
            ▼
      hidden_states
  """

  def __init__(self, config: unet_cfg.TransformerBlock2DConfig):
    """Initialize an instance of the TransformerBlock2D.

    Args:
      config (unet_cfg.TransformerBlock2Dconfig): the configuration of this
        block.
    """
    super().__init__()
    self.config = config
    self.pre_conv_norm = layers_builder.build_norm(
        config.attention_block_config.dim, config.pre_conv_normalization_config
    )
    self.conv_in = nn.Conv2d(
        config.attention_block_config.dim,
        config.attention_block_config.dim,
        kernel_size=1,
        padding=0,
    )
    self.self_attention = AttentionBlock2D(config.attention_block_config)
    self.cross_attention = CrossAttentionBlock2D(
        config.cross_attention_block_config
    )
    self.feed_forward = FeedForwardBlock2D(config.feed_forward_block_config)
    self.conv_out = nn.Conv2d(
        config.attention_block_config.dim,
        config.attention_block_config.dim,
        kernel_size=1,
        padding=0,
    )

  def forward(self, x: torch.Tensor, context: torch.Tensor):
    """Forward function of the TransformerBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      context_tensor (torch.Tensor): the context tensor to apply cross attention
        on.

    Returns:
      output activation tensor after transformer block.
    """
    residual_long = x

    x = self.pre_conv_norm(x)
    x = self.conv_in(x)
    x = self.self_attention(x)
    x = self.cross_attention(x, context)
    x = self.feed_forward(x)

    x = self.conv_out(x)
    x = x + residual_long

    return x


class DownEncoderBlock2D(nn.Module):
  """Encoder block containing several residual blocks with optional interleaved transformer blocks.

            input_tensor
                 |
  ┌──────────────▼─────────────┐
  │   ┌────────────────────┐   │
  │   │   ResidualBlock2D  │   │
  │   └──────────┬─────────┘   │
  │              │             │  num_layers
  │   ┌────────────────────┐   │
  │   │     (Optional)     │   │
  │   │ TransformerBlock2D │   │
  │   └──────────┬─────────┘   │
  └──────────────┬─────────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │     Downsampler    │
      └──────────┬─────────┘
                 │
                 ▼
           hidden_states
  """

  def __init__(self, config: unet_cfg.DownEncoderBlock2DConfig):
    """Initialize an instance of the DownEncoderBlock2D.

    Args:
      config (unet_cfg.DownEncoderBlock2DConfig): the configuration of this
        block.
    """
    super().__init__()
    self.config = config
    resnets = []
    transformers = []
    for i in range(config.num_layers):
      input_channels = config.in_channels if i == 0 else config.out_channels
      resnets.append(
          ResidualBlock2D(
              unet_cfg.ResidualBlock2DConfig(
                  in_channels=input_channels,
                  hidden_channels=config.out_channels,
                  out_channels=config.out_channels,
                  time_embedding_channels=config.time_embedding_channels,
                  normalization_config=config.normalization_config,
                  activation_config=config.activation_config,
              )
          )
      )
      if config.transformer_block_config:
        transformers.append(TransformerBlock2D(config.transformer_block_config))
    self.resnets = nn.ModuleList(resnets)
    self.transformers = (
        nn.ModuleList(transformers) if len(transformers) > 0 else None
    )
    if config.add_downsample:
      self.downsampler = unet_builder.build_downsampling(config.sampling_config)
    else:
      self.downsampler = None

  def forward(
      self,
      input_tensor: torch.Tensor,
      time_emb: Optional[torch.Tensor] = None,
      context_tensor: Optional[torch.Tensor] = None,
      output_hidden_states: bool = False,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Forward function of the DownEncoderBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is
        configured to accept time embedding.
      context_tensor (torch.Tensor): optional context tensor, if the block if
        configured to use transofrmer block.
      output_hidden_states (bool): whether to output hidden states, usually for
        skip connections.

    Returns:
      output hidden_states tensor after DownEncoderBlock2D.
    """
    hidden_states = input_tensor
    output_states = []
    for i, resnet in enumerate(self.resnets):
      hidden_states = resnet(hidden_states, time_emb)
      if self.transformers is not None:
        hidden_states = self.transformers[i](hidden_states, context_tensor)
      output_states.append(hidden_states)
    if self.downsampler:
      hidden_states = self.downsampler(hidden_states)
      output_states.append(hidden_states)
    if output_hidden_states:
      return hidden_states, output_states
    else:
      return hidden_states


class UpDecoderBlock2D(nn.Module):
  """Decoder block containing several residual blocks with optional interleaved transformer blocks.

            input_tensor
                 |
  ┌──────────────▼─────────────┐
  │   ┌────────────────────┐   │
  │   │   ResidualBlock2D  │   │
  │   └──────────┬─────────┘   │
  │              │             │  num_layers
  │   ┌────────────────────┐   │
  │   │     (Optional)     │   │
  │   │ TransformerBlock2D │   │
  │   └──────────┬─────────┘   │
  └──────────────┬─────────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │      Upsampler     │
      └──────────┬─────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │       Conv2D       │
      └──────────┬─────────┘
                 │
                 ▼
           hidden_states
  """

  def __init__(self, config: unet_cfg.UpDecoderBlock2DConfig):
    """Initialize an instance of the UpDecoderBlock2D.

    Args:
      config (unet_cfg.UpDecoderBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    resnets = []
    transformers = []
    for i in range(config.num_layers):
      input_channels = config.in_channels if i == 0 else config.out_channels
      resnets.append(
          ResidualBlock2D(
              unet_cfg.ResidualBlock2DConfig(
                  in_channels=input_channels,
                  hidden_channels=config.out_channels,
                  out_channels=config.out_channels,
                  time_embedding_channels=config.time_embedding_channels,
                  normalization_config=config.normalization_config,
                  activation_config=config.activation_config,
              )
          )
      )
      if config.transformer_block_config:
        transformers.append(TransformerBlock2D(config.transformer_block_config))
    self.resnets = nn.ModuleList(resnets)
    self.transformers = (
        nn.ModuleList(transformers) if len(transformers) > 0 else None
    )
    if config.add_upsample:
      self.upsampler = unet_builder.build_upsampling(config.sampling_config)
      if config.upsample_conv:
        self.upsample_conv = nn.Conv2d(
            config.out_channels,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    else:
      self.upsampler = None

  def forward(
      self,
      input_tensor: torch.Tensor,
      time_emb: Optional[torch.Tensor] = None,
      context_tensor: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the UpDecoderBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is
        configured to accept time embedding.
      context_tensor (torch.Tensor): optional context tensor, if the block if
        configured to use transofrmer block.

    Returns:
      output hidden_states tensor after UpDecoderBlock2D.
    """
    hidden_states = input_tensor
    for i, resnet in enumerate(self.resnets):
      hidden_states = resnet(hidden_states, time_emb)
      if self.transformers is not None:
        hidden_states = self.transformers[i](hidden_states, context_tensor)
    if self.upsampler:
      hidden_states = self.upsampler(hidden_states)
      if self.upsample_conv:
        hidden_states = self.upsample_conv(hidden_states)
    return hidden_states


class SkipUpDecoderBlock2D(nn.Module):
  """Decoder block contains skip connections and residual blocks with optional interleaved transformer blocks.

   input_tensor, skip_connection_tensors
                 |
  ┌──────────────▼─────────────┐
  │   ┌────────────────────┐   │
  │   │   ResidualBlock2D  │   │
  │   └──────────┬─────────┘   │
  │              │             │  num_layers
  │   ┌────────────────────┐   │
  │   │     (Optional)     │   │
  │   │ TransformerBlock2D │   │
  │   └──────────┬─────────┘   │
  └──────────────┬─────────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │      Upsampler     │
      └──────────┬─────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │       Conv2D       │
      └──────────┬─────────┘
                 │
                 ▼
           hidden_states
  """

  def __init__(self, config: unet_cfg.SkipUpDecoderBlock2DConfig):
    """Initialize an instance of the SkipUpDecoderBlock2D.

    Args:
      config (unet_cfg.SkipUpDecoderBlock2DConfig): the configuration of this
        block.
    """
    super().__init__()
    self.config = config
    resnets = []
    transformers = []
    for i in range(config.num_layers):
      res_skip_channels = (
          config.in_channels
          if (i == config.num_layers - 1)
          else config.out_channels
      )
      resnet_in_channels = (
          config.prev_out_channels if i == 0 else config.out_channels
      )
      resnets.append(
          ResidualBlock2D(
              unet_cfg.ResidualBlock2DConfig(
                  in_channels=resnet_in_channels + res_skip_channels,
                  hidden_channels=config.out_channels,
                  out_channels=config.out_channels,
                  time_embedding_channels=config.time_embedding_channels,
                  normalization_config=config.normalization_config,
                  activation_config=config.activation_config,
              )
          )
      )
      if config.transformer_block_config:
        transformers.append(TransformerBlock2D(config.transformer_block_config))
    self.resnets = nn.ModuleList(resnets)
    self.transformers = (
        nn.ModuleList(transformers) if len(transformers) > 0 else None
    )
    if config.add_upsample:
      self.upsampler = unet_builder.build_upsampling(config.sampling_config)
      if config.upsample_conv:
        self.upsample_conv = nn.Conv2d(
            config.out_channels,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    else:
      self.upsampler = None

  def forward(
      self,
      input_tensor: torch.Tensor,
      skip_connection_tensors: List[torch.Tensor],
      time_emb: Optional[torch.Tensor] = None,
      context_tensor: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the SkipUpDecoderBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      skip_connection_tensors (List[torch.Tensor]): the skip connection tensors
        from encoder blocks.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is
        configured to accept time embedding.
      context_tensor (torch.Tensor): optional context tensor, if the block if
        configured to use transofrmer block.

    Returns:
      output hidden_states tensor after SkipUpDecoderBlock2D.
    """
    hidden_states = input_tensor
    for i, (resnet, skip_connection_tensor) in enumerate(
        zip(self.resnets, skip_connection_tensors)
    ):
      hidden_states = torch.cat([hidden_states, skip_connection_tensor], dim=1)
      hidden_states = resnet(hidden_states, time_emb)
      if self.transformers is not None:
        hidden_states = self.transformers[i](hidden_states, context_tensor)
    if self.upsampler:
      hidden_states = self.upsampler(hidden_states)
      if self.upsample_conv:
        hidden_states = self.upsample_conv(hidden_states)
    return hidden_states


class MidBlock2D(nn.Module):
  """Middle block containing at least one residual blocks with optional interleaved attention blocks.

            input_tensor
                 |
                 ▼
       ┌───────────────────┐
       │  ResidualBlock2D  │
       └─────────┬─────────┘
                 │
  ┌──────────────▼─────────────┐
  │   ┌────────────────────┐   │
  │   │     (Optional)     │   │
  │   │  AttentionBlock2D  │   │
  │   └──────────┬─────────┘   │
  │              │             │
  │   ┌──────────▼─────────┐   │
  │   │     (Optional)     │   │  num_layers
  │   │ TransformerBlock2D │   │
  │   └──────────┬─────────┘   │
  │              │             │
  │   ┌──────────▼─────────┐   │
  │   │   ResidualBlock2D  │   │
  │   └────────────────────┘   │
  └──────────────┬─────────────┘
                 │
                 ▼
          hidden_states
  """

  def __init__(self, config: unet_cfg.MidBlock2DConfig):
    """Initialize an instance of the MidBlock2D.

    Args:
      config (unet_cfg.MidBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    resnets = [
        ResidualBlock2D(
            unet_cfg.ResidualBlock2DConfig(
                in_channels=config.in_channels,
                hidden_channels=config.in_channels,
                out_channels=config.in_channels,
                time_embedding_channels=config.time_embedding_channels,
                normalization_config=config.normalization_config,
                activation_config=config.activation_config,
            )
        )
    ]
    attentions = []
    transformers = []
    for i in range(config.num_layers):
      if self.config.attention_block_config:
        attentions.append(AttentionBlock2D(config.attention_block_config))
      if self.config.transformer_block_config:
        transformers.append(TransformerBlock2D(config.transformer_block_config))
      resnets.append(
          ResidualBlock2D(
              unet_cfg.ResidualBlock2DConfig(
                  in_channels=config.in_channels,
                  hidden_channels=config.in_channels,
                  out_channels=config.in_channels,
                  time_embedding_channels=config.time_embedding_channels,
                  normalization_config=config.normalization_config,
                  activation_config=config.activation_config,
              )
          )
      )
    self.resnets = nn.ModuleList(resnets)
    self.attentions = nn.ModuleList(attentions) if len(attentions) > 0 else None
    self.transformers = (
        nn.ModuleList(transformers) if len(transformers) > 0 else None
    )

  def forward(
      self,
      input_tensor: torch.Tensor,
      time_emb: Optional[torch.Tensor] = None,
      context_tensor: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the MidBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is
        configured to accept time embedding.
      context_tensor (torch.Tensor): optional context tensor, if the block if
        configured to use transofrmer block.

    Returns:
      output hidden_states tensor after MidBlock2D.
    """
    hidden_states = self.resnets[0](input_tensor, time_emb)
    for i, resnet in enumerate(self.resnets[1:]):
      if self.attentions is not None:
        hidden_states = self.attentions[i](hidden_states)
      if self.transformers is not None:
        hidden_states = self.transformers[i](hidden_states, context_tensor)
      hidden_states = resnet(hidden_states, time_emb)
    return hidden_states
