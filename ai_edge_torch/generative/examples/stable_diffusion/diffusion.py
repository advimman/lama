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

_down_encoder_blocks_tensor_names = [
    stable_diffusion_loader.DownEncoderBlockTensorNames(
        residual_block_tensor_names=[
            stable_diffusion_loader.ResidualBlockTensorNames(
                norm_1=f"model.diffusion_model.input_blocks.{i*3+j+1}.0.in_layers.0",
                conv_1=f"model.diffusion_model.input_blocks.{i*3+j+1}.0.in_layers.2",
                norm_2=f"model.diffusion_model.input_blocks.{i*3+j+1}.0.out_layers.0",
                conv_2=f"model.diffusion_model.input_blocks.{i*3+j+1}.0.out_layers.3",
                time_embedding=f"model.diffusion_model.input_blocks.{i*3+j+1}.0.emb_layers.1",
                residual_layer=f"model.diffusion_model.input_blocks.{i*3+j+1}.0.skip_connection"
                if (i * 3 + j + 1) in [4, 7]
                else None,
            )
            for j in range(2)
        ],
        transformer_block_tensor_names=[
            stable_diffusion_loader.TransformerBlockTensorNames(
                pre_conv_norm=(
                    f"model.diffusion_model.input_blocks.{i*3+j+1}.1.norm"
                ),
                conv_in=(
                    f"model.diffusion_model.input_blocks.{i*3+j+1}.1.proj_in"
                ),
                conv_out=(
                    f"model.diffusion_model.input_blocks.{i*3+j+1}.1.proj_out"
                ),
                self_attention=stable_diffusion_loader.AttentionBlockTensorNames(
                    norm=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.norm1",
                    q_proj=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.attn1.to_q",
                    k_proj=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.attn1.to_k",
                    v_proj=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.attn1.to_v",
                    output_proj=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.attn1.to_out.0",
                ),
                cross_attention=stable_diffusion_loader.CrossAttentionBlockTensorNames(
                    norm=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.norm2",
                    q_proj=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.attn2.to_q",
                    k_proj=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.attn2.to_k",
                    v_proj=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.attn2.to_v",
                    output_proj=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.attn2.to_out.0",
                ),
                feed_forward=stable_diffusion_loader.FeedForwardBlockTensorNames(
                    norm=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.norm3",
                    ge_glu=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.ff.net.0.proj",
                    w2=f"model.diffusion_model.input_blocks.{i*3+j+1}.1.transformer_blocks.0.ff.net.2",
                ),
            )
            for j in range(2)
        ]
        if i < 3
        else None,
        downsample_conv=f"model.diffusion_model.input_blocks.{i*3+3}.0.op"
        if i < 3
        else None,
    )
    for i in range(4)
]

_mid_block_tensor_names = stable_diffusion_loader.MidBlockTensorNames(
    residual_block_tensor_names=[
        stable_diffusion_loader.ResidualBlockTensorNames(
            norm_1=f"model.diffusion_model.middle_block.{i}.in_layers.0",
            conv_1=f"model.diffusion_model.middle_block.{i}.in_layers.2",
            norm_2=f"model.diffusion_model.middle_block.{i}.out_layers.0",
            conv_2=f"model.diffusion_model.middle_block.{i}.out_layers.3",
            time_embedding=(
                f"model.diffusion_model.middle_block.{i}.emb_layers.1"
            ),
        )
        for i in [0, 2]
    ],
    transformer_block_tensor_names=[
        stable_diffusion_loader.TransformerBlockTensorNames(
            pre_conv_norm=f"model.diffusion_model.middle_block.{i}.norm",
            conv_in=f"model.diffusion_model.middle_block.{i}.proj_in",
            conv_out=f"model.diffusion_model.middle_block.{i}.proj_out",
            self_attention=stable_diffusion_loader.AttentionBlockTensorNames(
                norm=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm1",
                q_proj=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_q",
                k_proj=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_k",
                v_proj=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_v",
                output_proj=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn1.to_out.0",
            ),
            cross_attention=stable_diffusion_loader.CrossAttentionBlockTensorNames(
                norm=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm2",
                q_proj=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_q",
                k_proj=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_k",
                v_proj=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_v",
                output_proj=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.attn2.to_out.0",
            ),
            feed_forward=stable_diffusion_loader.FeedForwardBlockTensorNames(
                norm=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.norm3",
                ge_glu=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.0.proj",
                w2=f"model.diffusion_model.middle_block.{i}.transformer_blocks.0.ff.net.2",
            ),
        )
        for i in [1]
    ],
)

_up_decoder_blocks_tensor_names = [
    stable_diffusion_loader.SkipUpDecoderBlockTensorNames(
        residual_block_tensor_names=[
            stable_diffusion_loader.ResidualBlockTensorNames(
                norm_1=(
                    f"model.diffusion_model.output_blocks.{i*3+j}.0.in_layers.0"
                ),
                conv_1=(
                    f"model.diffusion_model.output_blocks.{i*3+j}.0.in_layers.2"
                ),
                norm_2=f"model.diffusion_model.output_blocks.{i*3+j}.0.out_layers.0",
                conv_2=f"model.diffusion_model.output_blocks.{i*3+j}.0.out_layers.3",
                time_embedding=f"model.diffusion_model.output_blocks.{i*3+j}.0.emb_layers.1",
                residual_layer=f"model.diffusion_model.output_blocks.{i*3+j}.0.skip_connection",
            )
            for j in range(3)
        ],
        transformer_block_tensor_names=[
            stable_diffusion_loader.TransformerBlockTensorNames(
                pre_conv_norm=(
                    f"model.diffusion_model.output_blocks.{i*3+j}.1.norm"
                ),
                conv_in=(
                    f"model.diffusion_model.output_blocks.{i*3+j}.1.proj_in"
                ),
                conv_out=(
                    f"model.diffusion_model.output_blocks.{i*3+j}.1.proj_out"
                ),
                self_attention=stable_diffusion_loader.AttentionBlockTensorNames(
                    norm=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.norm1",
                    q_proj=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.attn1.to_q",
                    k_proj=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.attn1.to_k",
                    v_proj=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.attn1.to_v",
                    output_proj=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.attn1.to_out.0",
                ),
                cross_attention=stable_diffusion_loader.CrossAttentionBlockTensorNames(
                    norm=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.norm2",
                    q_proj=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.attn2.to_q",
                    k_proj=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.attn2.to_k",
                    v_proj=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.attn2.to_v",
                    output_proj=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.attn2.to_out.0",
                ),
                feed_forward=stable_diffusion_loader.FeedForwardBlockTensorNames(
                    norm=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.norm3",
                    ge_glu=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.ff.net.0.proj",
                    w2=f"model.diffusion_model.output_blocks.{i*3+j}.1.transformer_blocks.0.ff.net.2",
                ),
            )
            for j in range(3)
        ]
        if i > 0
        else None,
        upsample_conv=f"model.diffusion_model.output_blocks.{i*3+2}.2.conv"
        if 0 < i < 3
        else (
            f"model.diffusion_model.output_blocks.2.1.conv" if i == 0 else None
        ),
    )
    for i in range(4)
]

TENSOR_NAMES = stable_diffusion_loader.DiffusionModelLoader.TensorNames(
    time_embedding=stable_diffusion_loader.TimeEmbeddingTensorNames(
        w1="model.diffusion_model.time_embed.0",
        w2="model.diffusion_model.time_embed.2",
    ),
    conv_in="model.diffusion_model.input_blocks.0.0",
    conv_out="model.diffusion_model.out.2",
    final_norm="model.diffusion_model.out.0",
    down_encoder_blocks_tensor_names=_down_encoder_blocks_tensor_names,
    mid_block_tensor_names=_mid_block_tensor_names,
    up_decoder_blocks_tensor_names=_up_decoder_blocks_tensor_names,
)


def build_attention_config(
    num_heads,
    dim,
    num_query_groups,
    rotary_base=0,
    rotary_percentage=0.0,
    qkv_transpose_before_split=True,
    qkv_use_bias=False,
    output_proj_use_bias=True,
    enable_kv_cache=False,
    qkv_fused_interleaved=False,
):

  return layers_cfg.AttentionConfig(
      num_heads=num_heads,
      head_dim=dim // num_heads,
      num_query_groups=num_query_groups,
      rotary_base=rotary_base,
      rotary_percentage=rotary_percentage,
      qkv_transpose_before_split=qkv_transpose_before_split,
      qkv_use_bias=qkv_use_bias,
      output_proj_use_bias=output_proj_use_bias,
      enable_kv_cache=enable_kv_cache,
      qkv_fused_interleaved=qkv_fused_interleaved,
  )


class TimeEmbedding(nn.Module):

  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.w1 = nn.Linear(in_dim, out_dim)
    self.w2 = nn.Linear(out_dim, out_dim)
    self.act = layers_builder.get_activation(
        layers_cfg.ActivationConfig(layers_cfg.ActivationType.SILU)
    )

  def forward(self, x: torch.Tensor):
    return self.w2(self.act(self.w1(x)))


class Diffusion(nn.Module):
  """The Diffusion model used in Stable Diffusion.

    For details, see https://arxiv.org/abs/2103.00020

    Sturcture of the Diffusion model:

                         latents        text context   time embed
                            │                 │            │
                            │                 │            │
                  ┌─────────▼─────────┐       │  ┌─────────▼─────────┐
                  │      ConvIn       │       │  │   Time Embedding  │
                  └─────────┬─────────┘       │  └─────────┬─────────┘
                            │                 │            │
                  ┌─────────▼─────────┐       │            │
           ┌──────┤   DownEncoder2D   │ ◄─────┼────────────┤
           │      └─────────┬─────────┘ x 4   │            │
           │                │                 │            │
           │      ┌─────────▼─────────┐       │            │
  skip connection │     MidBlock2D    │ ◄─────┼────────────┤
           │      └─────────┬─────────┘       │            │
           │                │                 │            │
           │      ┌─────────▼─────────┐       │            │
           └──────►  SkipUpDecoder2D  │ ◄─────┴────────────┘
                  └─────────┬─────────┘ x 4
                            │
                  ┌─────────▼─────────┐
                  │     FinalNorm     │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │    Activation     │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │      ConvOut      │
                  └─────────┬─────────┘
                            │
                            ▼
                      output image
  """

  def __init__(self, config: unet_cfg.DiffusionModelConfig):
    super().__init__()

    self.config = config
    block_out_channels = config.block_out_channels
    reversed_block_out_channels = list(reversed(block_out_channels))

    time_embedding_blocks_dim = config.time_embedding_blocks_dim
    self.time_embedding = TimeEmbedding(
        config.time_embedding_dim, config.time_embedding_blocks_dim
    )

    self.conv_in = nn.Conv2d(
        config.in_channels, block_out_channels[0], kernel_size=3, padding=1
    )

    # Down encoders.
    down_encoders = []
    output_channel = block_out_channels[0]
    for i, block_out_channel in enumerate(block_out_channels):
      input_channel = output_channel
      output_channel = block_out_channel
      not_final_block = i < len(block_out_channels) - 1
      if not_final_block:
        down_encoders.append(
            blocks_2d.DownEncoderBlock2D(
                unet_cfg.DownEncoderBlock2DConfig(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    normalization_config=config.residual_norm_config,
                    activation_config=layers_cfg.ActivationConfig(
                        config.residual_activation_type
                    ),
                    num_layers=config.layers_per_block,
                    padding=config.downsample_padding,
                    time_embedding_channels=time_embedding_blocks_dim,
                    add_downsample=True,
                    sampling_config=unet_cfg.DownSamplingConfig(
                        mode=unet_cfg.SamplingType.CONVOLUTION,
                        in_channels=output_channel,
                        out_channels=output_channel,
                        kernel_size=3,
                        stride=2,
                        padding=config.downsample_padding,
                    ),
                    transformer_block_config=unet_cfg.TransformerBlock2DConfig(
                        attention_block_config=unet_cfg.AttentionBlock2DConfig(
                            dim=output_channel,
                            attention_batch_size=config.transformer_batch_size,
                            normalization_config=config.transformer_norm_config,
                            attention_config=build_attention_config(
                                num_heads=config.transformer_num_attention_heads,
                                dim=output_channel,
                                num_query_groups=config.transformer_num_attention_heads,
                            ),
                            enable_hlfb=config.enable_hlfb,
                        ),
                        cross_attention_block_config=unet_cfg.CrossAttentionBlock2DConfig(
                            query_dim=output_channel,
                            cross_dim=config.transformer_cross_attention_dim,
                            hidden_dim=output_channel,
                            output_dim=output_channel,
                            attention_batch_size=config.transformer_batch_size,
                            normalization_config=config.transformer_norm_config,
                            attention_config=build_attention_config(
                                num_heads=config.transformer_num_attention_heads,
                                dim=output_channel,
                                num_query_groups=config.transformer_num_attention_heads,
                            ),
                            enable_hlfb=config.enable_hlfb,
                        ),
                        pre_conv_normalization_config=config.transformer_pre_conv_norm_config,
                        feed_forward_block_config=unet_cfg.FeedForwardBlock2DConfig(
                            dim=output_channel,
                            hidden_dim=output_channel * 4,
                            normalization_config=config.transformer_norm_config,
                            activation_config=layers_cfg.ActivationConfig(
                                type=config.transformer_ff_activation_type,
                                dim_in=output_channel,
                                dim_out=output_channel * 4,
                            ),
                            use_bias=True,
                        ),
                    ),
                )
            )
        )
      else:
        down_encoders.append(
            blocks_2d.DownEncoderBlock2D(
                unet_cfg.DownEncoderBlock2DConfig(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    normalization_config=config.residual_norm_config,
                    activation_config=layers_cfg.ActivationConfig(
                        config.residual_activation_type
                    ),
                    num_layers=config.layers_per_block,
                    padding=config.downsample_padding,
                    time_embedding_channels=time_embedding_blocks_dim,
                    add_downsample=False,
                )
            )
        )
    self.down_encoders = nn.ModuleList(down_encoders)

    # Mid block.
    mid_block_channels = block_out_channels[-1]
    self.mid_block = blocks_2d.MidBlock2D(
        unet_cfg.MidBlock2DConfig(
            in_channels=block_out_channels[-1],
            normalization_config=config.residual_norm_config,
            activation_config=layers_cfg.ActivationConfig(
                config.residual_activation_type
            ),
            num_layers=config.mid_block_layers,
            time_embedding_channels=config.time_embedding_blocks_dim,
            transformer_block_config=unet_cfg.TransformerBlock2DConfig(
                attention_block_config=unet_cfg.AttentionBlock2DConfig(
                    dim=mid_block_channels,
                    attention_batch_size=config.transformer_batch_size,
                    normalization_config=config.transformer_norm_config,
                    attention_config=build_attention_config(
                        num_heads=config.transformer_num_attention_heads,
                        dim=mid_block_channels,
                        num_query_groups=config.transformer_num_attention_heads,
                    ),
                    enable_hlfb=config.enable_hlfb,
                ),
                cross_attention_block_config=unet_cfg.CrossAttentionBlock2DConfig(
                    query_dim=mid_block_channels,
                    cross_dim=config.transformer_cross_attention_dim,
                    hidden_dim=mid_block_channels,
                    output_dim=mid_block_channels,
                    attention_batch_size=config.transformer_batch_size,
                    normalization_config=config.transformer_norm_config,
                    attention_config=build_attention_config(
                        num_heads=config.transformer_num_attention_heads,
                        dim=mid_block_channels,
                        num_query_groups=config.transformer_num_attention_heads,
                    ),
                    enable_hlfb=config.enable_hlfb,
                ),
                pre_conv_normalization_config=config.transformer_pre_conv_norm_config,
                feed_forward_block_config=unet_cfg.FeedForwardBlock2DConfig(
                    dim=mid_block_channels,
                    hidden_dim=mid_block_channels * 4,
                    normalization_config=config.transformer_norm_config,
                    activation_config=layers_cfg.ActivationConfig(
                        type=config.transformer_ff_activation_type,
                        dim_in=mid_block_channels,
                        dim_out=mid_block_channels * 4,
                    ),
                    use_bias=True,
                ),
            ),
        )
    )

    # Up decoders.
    up_decoders = []
    up_decoder_layers_per_block = config.layers_per_block + 1
    output_channel = reversed_block_out_channels[0]
    for i, block_out_channel in enumerate(reversed_block_out_channels):
      prev_out_channel = output_channel
      output_channel = block_out_channel
      input_channel = reversed_block_out_channels[
          min(i + 1, len(reversed_block_out_channels) - 1)
      ]
      not_final_block = i < len(reversed_block_out_channels) - 1
      not_first_block = i != 0
      if not_first_block:
        up_decoders.append(
            blocks_2d.SkipUpDecoderBlock2D(
                unet_cfg.SkipUpDecoderBlock2DConfig(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_out_channels=prev_out_channel,
                    normalization_config=config.residual_norm_config,
                    activation_config=layers_cfg.ActivationConfig(
                        config.residual_activation_type
                    ),
                    num_layers=up_decoder_layers_per_block,
                    time_embedding_channels=time_embedding_blocks_dim,
                    add_upsample=not_final_block,
                    upsample_conv=True,
                    sampling_config=unet_cfg.UpSamplingConfig(
                        mode=unet_cfg.SamplingType.NEAREST,
                        scale_factor=2,
                    ),
                    transformer_block_config=unet_cfg.TransformerBlock2DConfig(
                        attention_block_config=unet_cfg.AttentionBlock2DConfig(
                            dim=output_channel,
                            attention_batch_size=config.transformer_batch_size,
                            normalization_config=config.transformer_norm_config,
                            attention_config=build_attention_config(
                                num_heads=config.transformer_num_attention_heads,
                                dim=output_channel,
                                num_query_groups=config.transformer_num_attention_heads,
                            ),
                            enable_hlfb=config.enable_hlfb,
                        ),
                        cross_attention_block_config=unet_cfg.CrossAttentionBlock2DConfig(
                            query_dim=output_channel,
                            cross_dim=config.transformer_cross_attention_dim,
                            hidden_dim=output_channel,
                            output_dim=output_channel,
                            attention_batch_size=config.transformer_batch_size,
                            normalization_config=config.transformer_norm_config,
                            attention_config=build_attention_config(
                                num_heads=config.transformer_num_attention_heads,
                                dim=output_channel,
                                num_query_groups=config.transformer_num_attention_heads,
                            ),
                            enable_hlfb=config.enable_hlfb,
                        ),
                        pre_conv_normalization_config=config.transformer_pre_conv_norm_config,
                        feed_forward_block_config=unet_cfg.FeedForwardBlock2DConfig(
                            dim=output_channel,
                            hidden_dim=output_channel * 4,
                            normalization_config=config.transformer_norm_config,
                            activation_config=layers_cfg.ActivationConfig(
                                type=config.transformer_ff_activation_type,
                                dim_in=output_channel,
                                dim_out=output_channel * 4,
                            ),
                            use_bias=True,
                        ),
                    ),
                )
            )
        )
      else:
        up_decoders.append(
            blocks_2d.SkipUpDecoderBlock2D(
                unet_cfg.SkipUpDecoderBlock2DConfig(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_out_channels=prev_out_channel,
                    normalization_config=config.residual_norm_config,
                    activation_config=layers_cfg.ActivationConfig(
                        config.residual_activation_type
                    ),
                    num_layers=up_decoder_layers_per_block,
                    time_embedding_channels=time_embedding_blocks_dim,
                    add_upsample=not_final_block,
                    upsample_conv=True,
                    sampling_config=unet_cfg.UpSamplingConfig(
                        mode=unet_cfg.SamplingType.NEAREST, scale_factor=2
                    ),
                )
            )
        )
    self.up_decoders = nn.ModuleList(up_decoders)

    self.final_norm = layers_builder.build_norm(
        reversed_block_out_channels[-1], config.final_norm_config
    )
    self.final_act = layers_builder.get_activation(
        layers_cfg.ActivationConfig(config.final_activation_type)
    )
    self.conv_out = nn.Conv2d(
        reversed_block_out_channels[-1],
        config.out_channels,
        kernel_size=3,
        padding=1,
    )

  @torch.inference_mode
  def forward(
      self, latents: torch.Tensor, context: torch.Tensor, time_emb: torch.Tensor
  ) -> torch.Tensor:
    """Forward function of diffusion model.

    Args:
        latents (torch.Tensor): latents space tensor.
        context (torch.Tensor): context tensor from CLIP text encoder.
        time_emb (torch.Tensor): the time embedding tensor.

    Returns:
        output latents from diffusion model.
    """
    time_emb = self.time_embedding(time_emb)
    x = self.conv_in(latents)
    skip_connection_tensors = [x]
    for encoder in self.down_encoders:
      x, hidden_states = encoder(
          x, time_emb, context, output_hidden_states=True
      )
      skip_connection_tensors.extend(hidden_states)
    x = self.mid_block(x, time_emb, context)
    for decoder in self.up_decoders:
      encoder_tensors = [
          skip_connection_tensors.pop()
          for i in range(self.config.layers_per_block + 1)
      ]
      x = decoder(x, encoder_tensors, time_emb, context)
    x = self.final_norm(x)
    x = self.final_act(x)
    x = self.conv_out(x)
    return x


def get_model_config(
    batch_size: int, device_type: str = "cpu"
) -> unet_cfg.DiffusionModelConfig:
  """Get configs for the Diffusion model of Stable Diffusion v1.5.

  Args:
    batch_size (int): the batch size of input.
    device_type (str): the device type of the model. Default to "cpu".

  Returns:
    The configuration of diffusion model of Stable Diffusion v1.5.
  """
  in_channels = 4
  out_channels = 4
  block_out_channels = [320, 640, 1280, 1280]
  layers_per_block = 2
  downsample_padding = 1

  # For now, only turns on StableHLO composite ops on GPU backend for better
  # performance. CPU should also switch to it once the support is done.
  enable_hlfb = True if device_type == "gpu" else False

  # Residual configs.
  residual_norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.GROUP_NORM,
      group_num=32,
      enable_hlfb=enable_hlfb,
  )
  residual_activation_type = layers_cfg.ActivationType.SILU

  # Transformer configs.
  transformer_num_attention_heads = 8
  transformer_batch_size = batch_size
  transformer_cross_attention_dim = 768  # Embedding from CLIP model
  transformer_pre_conv_norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.GROUP_NORM,
      epsilon=1e-6,
      group_num=32,
      enable_hlfb=enable_hlfb,
  )
  transformer_norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.LAYER_NORM,
      enable_hlfb=enable_hlfb,
  )
  transformer_ff_activation_type = layers_cfg.ActivationType.GE_GLU

  # Time embedding configs.
  time_embedding_dim = 320
  time_embedding_blocks_dim = 1280

  # Mid block configs.
  mid_block_layers = 1

  # Finaly layer configs.
  final_norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.GROUP_NORM,
      group_num=32,
      enable_hlfb=enable_hlfb,
  )
  final_activation_type = layers_cfg.ActivationType.SILU

  return unet_cfg.DiffusionModelConfig(
      in_channels=in_channels,
      out_channels=out_channels,
      block_out_channels=block_out_channels,
      layers_per_block=layers_per_block,
      downsample_padding=downsample_padding,
      residual_norm_config=residual_norm_config,
      residual_activation_type=residual_activation_type,
      transformer_batch_size=transformer_batch_size,
      transformer_num_attention_heads=transformer_num_attention_heads,
      transformer_cross_attention_dim=transformer_cross_attention_dim,
      transformer_pre_conv_norm_config=transformer_pre_conv_norm_config,
      transformer_norm_config=transformer_norm_config,
      transformer_ff_activation_type=transformer_ff_activation_type,
      mid_block_layers=mid_block_layers,
      time_embedding_dim=time_embedding_dim,
      time_embedding_blocks_dim=time_embedding_blocks_dim,
      final_norm_config=final_norm_config,
      final_activation_type=final_activation_type,
      enable_hlfb=enable_hlfb,
  )


def get_fake_model_config(
    batch_size: int, device_type: str = "cpu"
) -> unet_cfg.DiffusionModelConfig:
  """Get fake configs for the Diffusion model of Stable Diffusion v1.5 for testing.

  Args:
    batch_size (int): the batch size of input.
    device_type (str): the device type of the model. Default to "cpu".

  Returns:
    The configuration of diffusion model of Stable Diffusion v1.5.
  """
  in_channels = 4
  out_channels = 4
  block_out_channels = [2, 4, 8, 8]
  layers_per_block = 1
  downsample_padding = 1

  # For now, only turns on StableHLO composite ops on GPU backend for better
  # performance. CPU should also switch to it once the support is done.
  enable_hlfb = True if device_type == "gpu" else False

  # Residual configs.
  residual_norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.GROUP_NORM,
      group_num=2,
      enable_hlfb=enable_hlfb,
  )
  residual_activation_type = layers_cfg.ActivationType.SILU

  # Transformer configs.
  transformer_num_attention_heads = 1
  transformer_batch_size = batch_size
  transformer_cross_attention_dim = 4  # Embedding from CLIP model
  transformer_pre_conv_norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.GROUP_NORM,
      epsilon=1e-6,
      group_num=2,
      enable_hlfb=enable_hlfb,
  )
  transformer_norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.LAYER_NORM,
      enable_hlfb=enable_hlfb,
  )
  transformer_ff_activation_type = layers_cfg.ActivationType.GE_GLU

  # Time embedding configs.
  time_embedding_dim = 2
  time_embedding_blocks_dim = 4

  # Mid block configs.
  mid_block_layers = 1

  # Finaly layer configs.
  final_norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.GROUP_NORM,
      group_num=2,
      enable_hlfb=enable_hlfb,
  )
  final_activation_type = layers_cfg.ActivationType.SILU

  return unet_cfg.DiffusionModelConfig(
      in_channels=in_channels,
      out_channels=out_channels,
      block_out_channels=block_out_channels,
      layers_per_block=layers_per_block,
      downsample_padding=downsample_padding,
      residual_norm_config=residual_norm_config,
      residual_activation_type=residual_activation_type,
      transformer_batch_size=transformer_batch_size,
      transformer_num_attention_heads=transformer_num_attention_heads,
      transformer_cross_attention_dim=transformer_cross_attention_dim,
      transformer_pre_conv_norm_config=transformer_pre_conv_norm_config,
      transformer_norm_config=transformer_norm_config,
      transformer_ff_activation_type=transformer_ff_activation_type,
      mid_block_layers=mid_block_layers,
      time_embedding_dim=time_embedding_dim,
      time_embedding_blocks_dim=time_embedding_blocks_dim,
      final_norm_config=final_norm_config,
      final_activation_type=final_activation_type,
      enable_hlfb=enable_hlfb,
  )
