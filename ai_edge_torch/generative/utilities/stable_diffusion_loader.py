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
# Common utility functions for data loading etc.
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import ai_edge_torch.generative.layers.model_config as layers_config
import ai_edge_torch.generative.layers.unet.model_config as unet_config
import ai_edge_torch.generative.utilities.loader as loader
import torch


@dataclass
class ResidualBlockTensorNames:
  norm_1: str = None
  conv_1: str = None
  norm_2: str = None
  conv_2: str = None
  residual_layer: str = None
  time_embedding: str = None


@dataclass
class AttentionBlockTensorNames:
  norm: str = None
  fused_qkv_proj: str = None
  q_proj: str = None
  k_proj: str = None
  v_proj: str = None
  output_proj: str = None


@dataclass
class CrossAttentionBlockTensorNames:
  norm: str = None
  q_proj: str = None
  k_proj: str = None
  v_proj: str = None
  output_proj: str = None


@dataclass
class TimeEmbeddingTensorNames:
  w1: str = None
  w2: str = None


@dataclass
class FeedForwardBlockTensorNames:
  w1: str = None
  w2: str = None
  norm: str = None
  ge_glu: str = None


@dataclass
class TransformerBlockTensorNames:
  pre_conv_norm: str
  conv_in: str
  self_attention: AttentionBlockTensorNames
  cross_attention: CrossAttentionBlockTensorNames
  feed_forward: FeedForwardBlockTensorNames
  conv_out: str


@dataclass
class MidBlockTensorNames:
  residual_block_tensor_names: List[ResidualBlockTensorNames]
  attention_block_tensor_names: Optional[List[AttentionBlockTensorNames]] = None
  transformer_block_tensor_names: Optional[
      List[TransformerBlockTensorNames]
  ] = None


@dataclass
class DownEncoderBlockTensorNames:
  residual_block_tensor_names: List[ResidualBlockTensorNames]
  transformer_block_tensor_names: Optional[
      List[TransformerBlockTensorNames]
  ] = None
  downsample_conv: str = None


@dataclass
class UpDecoderBlockTensorNames:
  residual_block_tensor_names: List[ResidualBlockTensorNames]
  transformer_block_tensor_names: Optional[
      List[TransformerBlockTensorNames]
  ] = None
  upsample_conv: str = None


@dataclass
class SkipUpDecoderBlockTensorNames:
  residual_block_tensor_names: List[ResidualBlockTensorNames]
  transformer_block_tensor_names: Optional[
      List[TransformerBlockTensorNames]
  ] = None
  upsample_conv: str = None


def _map_to_converted_state(
    state: Dict[str, torch.Tensor],
    state_param: str,
    converted_state: Dict[str, torch.Tensor],
    converted_state_param: str,
    squeeze_dims: bool = False,
):
  converted_state[f"{converted_state_param}.weight"] = state.pop(
      f"{state_param}.weight"
  )
  if squeeze_dims:
    converted_state[f"{converted_state_param}.weight"] = torch.squeeze(
        converted_state[f"{converted_state_param}.weight"]
    )
  if f"{state_param}.bias" in state:
    converted_state[f"{converted_state_param}.bias"] = state.pop(
        f"{state_param}.bias"
    )
    if squeeze_dims:
      converted_state[f"{converted_state_param}.bias"] = torch.squeeze(
          converted_state[f"{converted_state_param}.bias"]
      )


class BaseLoader(loader.ModelLoader):

  def _map_residual_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      tensor_names: ResidualBlockTensorNames,
      converted_state_param_prefix: str,
      config: unet_config.ResidualBlock2DConfig,
  ):
    _map_to_converted_state(
        state,
        tensor_names.norm_1,
        converted_state,
        f"{converted_state_param_prefix}.norm_1",
    )
    _map_to_converted_state(
        state,
        tensor_names.conv_1,
        converted_state,
        f"{converted_state_param_prefix}.conv_1",
    )
    _map_to_converted_state(
        state,
        tensor_names.norm_2,
        converted_state,
        f"{converted_state_param_prefix}.norm_2",
    )
    _map_to_converted_state(
        state,
        tensor_names.conv_2,
        converted_state,
        f"{converted_state_param_prefix}.conv_2",
    )
    if config.in_channels != config.out_channels:
      _map_to_converted_state(
          state,
          tensor_names.residual_layer,
          converted_state,
          f"{converted_state_param_prefix}.residual_layer",
      )
    if config.time_embedding_channels is not None:
      _map_to_converted_state(
          state,
          tensor_names.time_embedding,
          converted_state,
          f"{converted_state_param_prefix}.time_emb_proj",
      )

  def _map_attention_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      tensor_names: AttentionBlockTensorNames,
      converted_state_param_prefix: str,
      config: unet_config.AttentionBlock2DConfig,
  ):
    if config.normalization_config.type != layers_config.NormalizationType.NONE:
      _map_to_converted_state(
          state,
          tensor_names.norm,
          converted_state,
          f"{converted_state_param_prefix}.norm",
      )
    attention_layer_prefix = f"{converted_state_param_prefix}.attention"
    if tensor_names.fused_qkv_proj is not None:
      _map_to_converted_state(
          state,
          tensor_names.fused_qkv_proj,
          converted_state,
          f"{attention_layer_prefix}.qkv_projection",
      )
    else:
      _map_to_converted_state(
          state,
          tensor_names.q_proj,
          converted_state,
          f"{attention_layer_prefix}.q_projection",
          squeeze_dims=True,
      )
      _map_to_converted_state(
          state,
          tensor_names.k_proj,
          converted_state,
          f"{attention_layer_prefix}.k_projection",
          squeeze_dims=True,
      )
      _map_to_converted_state(
          state,
          tensor_names.v_proj,
          converted_state,
          f"{attention_layer_prefix}.v_projection",
          squeeze_dims=True,
      )
      converted_state[f"{attention_layer_prefix}.qkv_projection.weight"] = (
          torch.concat(
              [
                  converted_state[
                      f"{attention_layer_prefix}.q_projection.weight"
                  ],
                  converted_state[
                      f"{attention_layer_prefix}.k_projection.weight"
                  ],
                  converted_state[
                      f"{attention_layer_prefix}.v_projection.weight"
                  ],
              ],
              axis=0,
          )
      )
      del converted_state[f"{attention_layer_prefix}.q_projection.weight"]
      del converted_state[f"{attention_layer_prefix}.k_projection.weight"]
      del converted_state[f"{attention_layer_prefix}.v_projection.weight"]
      if config.attention_config.qkv_use_bias:
        converted_state[f"{attention_layer_prefix}.qkv_projection.bias"] = (
            torch.concat(
                [
                    converted_state[
                        f"{attention_layer_prefix}.q_projection.bias"
                    ],
                    converted_state[
                        f"{attention_layer_prefix}.k_projection.bias"
                    ],
                    converted_state[
                        f"{attention_layer_prefix}.v_projection.bias"
                    ],
                ],
                axis=0,
            )
        )
        del converted_state[f"{attention_layer_prefix}.q_projection.bias"]
        del converted_state[f"{attention_layer_prefix}.k_projection.bias"]
        del converted_state[f"{attention_layer_prefix}.v_projection.bias"]

    _map_to_converted_state(
        state,
        tensor_names.output_proj,
        converted_state,
        f"{attention_layer_prefix}.output_projection",
        squeeze_dims=True,
    )

  def _map_cross_attention_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      tensor_names: CrossAttentionBlockTensorNames,
      converted_state_param_prefix: str,
      config: unet_config.CrossAttentionBlock2DConfig,
  ):
    if config.normalization_config.type != layers_config.NormalizationType.NONE:
      _map_to_converted_state(
          state,
          tensor_names.norm,
          converted_state,
          f"{converted_state_param_prefix}.norm",
      )
    attention_layer_prefix = f"{converted_state_param_prefix}.attention"
    _map_to_converted_state(
        state,
        tensor_names.q_proj,
        converted_state,
        f"{attention_layer_prefix}.q_projection",
    )
    _map_to_converted_state(
        state,
        tensor_names.k_proj,
        converted_state,
        f"{attention_layer_prefix}.k_projection",
    )
    _map_to_converted_state(
        state,
        tensor_names.v_proj,
        converted_state,
        f"{attention_layer_prefix}.v_projection",
    )
    _map_to_converted_state(
        state,
        tensor_names.output_proj,
        converted_state,
        f"{attention_layer_prefix}.output_projection",
    )

  def _map_feedforward_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      tensor_names: FeedForwardBlockTensorNames,
      converted_state_param_prefix: str,
      config: unet_config.FeedForwardBlock2DConfig,
  ):
    _map_to_converted_state(
        state,
        tensor_names.norm,
        converted_state,
        f"{converted_state_param_prefix}.norm",
    )
    if config.activation_config.type == layers_config.ActivationType.GE_GLU:
      _map_to_converted_state(
          state,
          tensor_names.ge_glu,
          converted_state,
          f"{converted_state_param_prefix}.act.proj",
      )
    else:
      _map_to_converted_state(
          state,
          tensor_names.w1,
          converted_state,
          f"{converted_state_param_prefix}.w1",
      )

    _map_to_converted_state(
        state,
        tensor_names.w2,
        converted_state,
        f"{converted_state_param_prefix}.w2",
    )

  def _map_transformer_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      tensor_names: TransformerBlockTensorNames,
      converted_state_param_prefix: str,
      config: unet_config.TransformerBlock2DConfig,
  ):
    _map_to_converted_state(
        state,
        tensor_names.pre_conv_norm,
        converted_state,
        f"{converted_state_param_prefix}.pre_conv_norm",
    )
    _map_to_converted_state(
        state,
        tensor_names.conv_in,
        converted_state,
        f"{converted_state_param_prefix}.conv_in",
    )
    self._map_attention_block(
        state,
        converted_state,
        tensor_names.self_attention,
        f"{converted_state_param_prefix}.self_attention",
        config.attention_block_config,
    )
    self._map_cross_attention_block(
        state,
        converted_state,
        tensor_names.cross_attention,
        f"{converted_state_param_prefix}.cross_attention",
        config.cross_attention_block_config,
    )
    self._map_feedforward_block(
        state,
        converted_state,
        tensor_names.feed_forward,
        f"{converted_state_param_prefix}.feed_forward",
        config.feed_forward_block_config,
    )
    _map_to_converted_state(
        state,
        tensor_names.conv_out,
        converted_state,
        f"{converted_state_param_prefix}.conv_out",
    )

  def _map_mid_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      tensor_names: MidBlockTensorNames,
      converted_state_param_prefix: str,
      config: unet_config.MidBlock2DConfig,
  ):
    residual_block_config = unet_config.ResidualBlock2DConfig(
        in_channels=config.in_channels,
        hidden_channels=config.in_channels,
        out_channels=config.in_channels,
        time_embedding_channels=config.time_embedding_channels,
        normalization_config=config.normalization_config,
        activation_config=config.activation_config,
    )
    self._map_residual_block(
        state,
        converted_state,
        tensor_names.residual_block_tensor_names[0],
        f"{converted_state_param_prefix}.resnets.0",
        residual_block_config,
    )
    for i in range(config.num_layers):
      if config.attention_block_config:
        self._map_attention_block(
            state,
            converted_state,
            tensor_names.attention_block_tensor_names[i],
            f"{converted_state_param_prefix}.attentions.{i}",
            config.attention_block_config,
        )
      if config.transformer_block_config:
        self._map_transformer_block(
            state,
            converted_state,
            tensor_names.transformer_block_tensor_names[i],
            f"{converted_state_param_prefix}.transformers.{i}",
            config.transformer_block_config,
        )
      self._map_residual_block(
          state,
          converted_state,
          tensor_names.residual_block_tensor_names[i + 1],
          f"{converted_state_param_prefix}.resnets.{i+1}",
          residual_block_config,
      )

  def _map_down_encoder_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      converted_state_param_prefix: str,
      config: unet_config.DownEncoderBlock2DConfig,
      tensor_names: DownEncoderBlockTensorNames,
  ):
    for i in range(config.num_layers):
      input_channels = config.in_channels if i == 0 else config.out_channels
      self._map_residual_block(
          state,
          converted_state,
          tensor_names.residual_block_tensor_names[i],
          f"{converted_state_param_prefix}.resnets.{i}",
          unet_config.ResidualBlock2DConfig(
              in_channels=input_channels,
              hidden_channels=config.out_channels,
              out_channels=config.out_channels,
              time_embedding_channels=config.time_embedding_channels,
              normalization_config=config.normalization_config,
              activation_config=config.activation_config,
          ),
      )
      if config.transformer_block_config:
        self._map_transformer_block(
            state,
            converted_state,
            tensor_names.transformer_block_tensor_names[i],
            f"{converted_state_param_prefix}.transformers.{i}",
            config.transformer_block_config,
        )
    if (
        config.add_downsample
        and config.sampling_config.mode == unet_config.SamplingType.CONVOLUTION
    ):
      _map_to_converted_state(
          state,
          tensor_names.downsample_conv,
          converted_state,
          f"{converted_state_param_prefix}.downsampler",
      )

  def _map_up_decoder_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      converted_state_param_prefix: str,
      config: unet_config.UpDecoderBlock2DConfig,
      tensor_names: UpDecoderBlockTensorNames,
  ):
    for i in range(config.num_layers):
      input_channels = config.in_channels if i == 0 else config.out_channels
      self._map_residual_block(
          state,
          converted_state,
          tensor_names.residual_block_tensor_names[i],
          f"{converted_state_param_prefix}.resnets.{i}",
          unet_config.ResidualBlock2DConfig(
              in_channels=input_channels,
              hidden_channels=config.out_channels,
              out_channels=config.out_channels,
              time_embedding_channels=config.time_embedding_channels,
              normalization_config=config.normalization_config,
              activation_config=config.activation_config,
          ),
      )
      if config.transformer_block_config:
        self._map_transformer_block(
            state,
            converted_state,
            tensor_names.transformer_block_tensor_names[i],
            f"{converted_state_param_prefix}.transformers.{i}",
            config.transformer_block_config,
        )
    if config.add_upsample and config.upsample_conv:
      _map_to_converted_state(
          state,
          tensor_names.upsample_conv,
          converted_state,
          f"{converted_state_param_prefix}.upsample_conv",
      )

  def _map_skip_up_decoder_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      converted_state_param_prefix: str,
      config: unet_config.SkipUpDecoderBlock2DConfig,
      tensor_names: UpDecoderBlockTensorNames,
  ):
    for i in range(config.num_layers):
      res_skip_channels = (
          config.in_channels
          if (i == config.num_layers - 1)
          else config.out_channels
      )
      resnet_in_channels = (
          config.prev_out_channels if i == 0 else config.out_channels
      )
      self._map_residual_block(
          state,
          converted_state,
          tensor_names.residual_block_tensor_names[i],
          f"{converted_state_param_prefix}.resnets.{i}",
          unet_config.ResidualBlock2DConfig(
              in_channels=resnet_in_channels + res_skip_channels,
              hidden_channels=config.out_channels,
              out_channels=config.out_channels,
              time_embedding_channels=config.time_embedding_channels,
              normalization_config=config.normalization_config,
              activation_config=config.activation_config,
          ),
      )
      if config.transformer_block_config:
        self._map_transformer_block(
            state,
            converted_state,
            tensor_names.transformer_block_tensor_names[i],
            f"{converted_state_param_prefix}.transformers.{i}",
            config.transformer_block_config,
        )
    if config.add_upsample and config.upsample_conv:
      _map_to_converted_state(
          state,
          tensor_names.upsample_conv,
          converted_state,
          f"{converted_state_param_prefix}.upsample_conv",
      )


# Alias class name for better code reading.
ClipModelLoader = BaseLoader


class AutoEncoderModelLoader(BaseLoader):

  @dataclass
  class TensorNames:
    quant_conv: str = None
    post_quant_conv: str = None
    conv_in: str = None
    conv_out: str = None
    final_norm: str = None
    mid_block_tensor_names: MidBlockTensorNames = None
    up_decoder_blocks_tensor_names: List[UpDecoderBlockTensorNames] = None

  def __init__(self, file_name: str, names: TensorNames):
    """AutoEncoderModelLoader constructor.

    Can be used to load encoder and decoder models.

    Args:
        file_name (str): Path to the checkpoint. Can be a directory or an exact
          file.
        names (TensorNames): An instance of `TensorNames` to determine mappings.
    """
    self._file_name = file_name
    self._names = names
    self._loader = self._get_loader()

  def load(
      self, model: torch.nn.Module, strict: bool = True
  ) -> Tuple[List[str], List[str]]:
    """Load the model from the checkpoint.

    Args:
        model (torch.nn.Module): The pytorch model that needs to be loaded.
        strict (bool, optional): Whether the converted keys are strictly
          matched. Defaults to True.

    Returns:
        missing_keys (List[str]): a list of str containing the missing keys.
        unexpected_keys (List[str]): a list of str containing the unexpected
        keys.

    Raises:
        ValueError: If conversion results in unmapped tensors and strict mode is
          enabled.
    """
    state = self._loader(self._file_name)
    converted_state = dict()
    if self._names.quant_conv is not None:
      _map_to_converted_state(
          state, self._names.quant_conv, converted_state, "quant_conv"
      )
    if self._names.post_quant_conv is not None:
      _map_to_converted_state(
          state, self._names.post_quant_conv, converted_state, "post_quant_conv"
      )
    if self._names.conv_in is not None:
      _map_to_converted_state(
          state, self._names.conv_in, converted_state, "conv_in"
      )
    if self._names.conv_out is not None:
      _map_to_converted_state(
          state, self._names.conv_out, converted_state, "conv_out"
      )
    if self._names.final_norm is not None:
      _map_to_converted_state(
          state, self._names.final_norm, converted_state, "final_norm"
      )
    self._map_mid_block(
        state,
        converted_state,
        self._names.mid_block_tensor_names,
        "mid_block",
        model.config.mid_block_config,
    )

    reversed_block_out_channels = list(
        reversed(model.config.block_out_channels)
    )
    block_out_channels = reversed_block_out_channels[0]
    for i, out_channels in enumerate(reversed_block_out_channels):
      prev_output_channel = block_out_channels
      block_out_channels = out_channels
      not_final_block = i < len(reversed_block_out_channels) - 1
      self._map_up_decoder_block(
          state,
          converted_state,
          f"up_decoder_blocks.{i}",
          unet_config.UpDecoderBlock2DConfig(
              in_channels=prev_output_channel,
              out_channels=block_out_channels,
              normalization_config=model.config.normalization_config,
              activation_config=model.config.activation_config,
              num_layers=model.config.layers_per_block,
              add_upsample=not_final_block,
              upsample_conv=True,
          ),
          self._names.up_decoder_blocks_tensor_names[i],
      )
    if strict and state:
      raise ValueError(
          f"Failed to map all tensor. Remaing tensor are: {list(state.keys())}"
      )
    return model.load_state_dict(converted_state, strict=strict)


def build_attention_config(
    num_heads,
    dim,
    num_query_groups,
    rotary_percentage=0.0,
    qkv_transpose_before_split=True,
    qkv_use_bias=False,
    output_proj_use_bias=True,
    enable_kv_cache=False,
    qkv_fused_interleaved=False,
):

  return layers_config.AttentionConfig(
      num_heads=num_heads,
      head_dim=dim // num_heads,
      num_query_groups=num_query_groups,
      rotary_percentage=rotary_percentage,
      qkv_transpose_before_split=qkv_transpose_before_split,
      qkv_use_bias=qkv_use_bias,
      output_proj_use_bias=output_proj_use_bias,
      enable_kv_cache=enable_kv_cache,
      qkv_fused_interleaved=qkv_fused_interleaved,
  )


class DiffusionModelLoader(BaseLoader):

  @dataclass
  class TensorNames:
    time_embedding: TimeEmbeddingTensorNames = None
    conv_in: str = None
    conv_out: str = None
    final_norm: str = None
    down_encoder_blocks_tensor_names: List[DownEncoderBlockTensorNames] = None
    mid_block_tensor_names: MidBlockTensorNames = None
    up_decoder_blocks_tensor_names: List[UpDecoderBlockTensorNames] = None

  def __init__(self, file_name: str, names: TensorNames):
    """DiffusionModelLoader constructor.

    Can be used to load diffusion models of Stable Diffusion.

    Args:
        file_name (str): Path to the checkpoint. Can be a directory or an exact
          file.
        names (TensorNames): An instance of `TensorNames` to determine mappings.
    """
    self._file_name = file_name
    self._names = names
    self._loader = self._get_loader()

  def load(
      self, model: torch.nn.Module, strict: bool = True
  ) -> Tuple[List[str], List[str]]:
    """Load the model from the checkpoint.

    Args:
        model (torch.nn.Module): The pytorch model that needs to be loaded.
        strict (bool, optional): Whether the converted keys are strictly
          matched. Defaults to True.

    Returns:
        missing_keys (List[str]): a list of str containing the missing keys.
        unexpected_keys (List[str]): a list of str containing the unexpected
        keys.

    Raises:
        ValueError: If conversion results in unmapped tensors and strict mode is
          enabled.
    """
    state = self._loader(self._file_name)
    converted_state = dict()
    config: unet_config.DiffusionModelConfig = model.config
    self._map_time_embedding(
        state, converted_state, "time_embedding", self._names.time_embedding
    )
    _map_to_converted_state(
        state, self._names.conv_in, converted_state, "conv_in"
    )
    _map_to_converted_state(
        state, self._names.conv_out, converted_state, "conv_out"
    )
    _map_to_converted_state(
        state, self._names.final_norm, converted_state, "final_norm"
    )

    # Map down_encoders.
    output_channel = config.block_out_channels[0]
    for i, block_out_channel in enumerate(config.block_out_channels):
      input_channel = output_channel
      output_channel = block_out_channel
      not_final_block = i < len(config.block_out_channels) - 1
      if not_final_block:
        down_encoder_block_config = unet_config.DownEncoderBlock2DConfig(
            in_channels=input_channel,
            out_channels=output_channel,
            normalization_config=config.residual_norm_config,
            activation_config=layers_config.ActivationConfig(
                config.residual_activation_type
            ),
            num_layers=config.layers_per_block,
            padding=config.downsample_padding,
            time_embedding_channels=config.time_embedding_blocks_dim,
            add_downsample=True,
            sampling_config=unet_config.DownSamplingConfig(
                mode=unet_config.SamplingType.CONVOLUTION,
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=2,
                padding=config.downsample_padding,
            ),
            transformer_block_config=unet_config.TransformerBlock2DConfig(
                attention_block_config=unet_config.AttentionBlock2DConfig(
                    dim=output_channel,
                    normalization_config=config.transformer_norm_config,
                    attention_config=build_attention_config(
                        num_heads=config.transformer_num_attention_heads,
                        dim=output_channel,
                        num_query_groups=config.transformer_num_attention_heads,
                    ),
                ),
                cross_attention_block_config=unet_config.CrossAttentionBlock2DConfig(
                    query_dim=output_channel,
                    cross_dim=config.transformer_cross_attention_dim,
                    hidden_dim=output_channel,
                    output_dim=output_channel,
                    normalization_config=config.transformer_norm_config,
                    attention_config=build_attention_config(
                        num_heads=config.transformer_num_attention_heads,
                        dim=output_channel,
                        num_query_groups=config.transformer_num_attention_heads,
                    ),
                ),
                pre_conv_normalization_config=config.transformer_pre_conv_norm_config,
                feed_forward_block_config=unet_config.FeedForwardBlock2DConfig(
                    dim=output_channel,
                    hidden_dim=output_channel * 4,
                    normalization_config=config.transformer_norm_config,
                    activation_config=layers_config.ActivationConfig(
                        type=config.transformer_ff_activation_type,
                        dim_in=output_channel,
                        dim_out=output_channel * 4,
                    ),
                    use_bias=True,
                ),
            ),
        )
      else:
        down_encoder_block_config = unet_config.DownEncoderBlock2DConfig(
            in_channels=input_channel,
            out_channels=output_channel,
            normalization_config=config.residual_norm_config,
            activation_config=layers_config.ActivationConfig(
                config.residual_activation_type
            ),
            num_layers=config.layers_per_block,
            padding=config.downsample_padding,
            time_embedding_channels=config.time_embedding_blocks_dim,
            add_downsample=False,
        )

      self._map_down_encoder_block(
          state,
          converted_state,
          f"down_encoders.{i}",
          down_encoder_block_config,
          self._names.down_encoder_blocks_tensor_names[i],
      )

    # Map mid block.
    mid_block_channels = config.block_out_channels[-1]
    mid_block_config = unet_config.MidBlock2DConfig(
        in_channels=mid_block_channels,
        normalization_config=config.residual_norm_config,
        activation_config=layers_config.ActivationConfig(
            config.residual_activation_type
        ),
        num_layers=config.mid_block_layers,
        time_embedding_channels=config.time_embedding_blocks_dim,
        transformer_block_config=unet_config.TransformerBlock2DConfig(
            attention_block_config=unet_config.AttentionBlock2DConfig(
                dim=mid_block_channels,
                normalization_config=config.transformer_norm_config,
                attention_config=build_attention_config(
                    num_heads=config.transformer_num_attention_heads,
                    dim=mid_block_channels,
                    num_query_groups=config.transformer_num_attention_heads,
                ),
            ),
            cross_attention_block_config=unet_config.CrossAttentionBlock2DConfig(
                query_dim=mid_block_channels,
                cross_dim=config.transformer_cross_attention_dim,
                hidden_dim=mid_block_channels,
                output_dim=mid_block_channels,
                normalization_config=config.transformer_norm_config,
                attention_config=build_attention_config(
                    num_heads=config.transformer_num_attention_heads,
                    dim=mid_block_channels,
                    num_query_groups=config.transformer_num_attention_heads,
                ),
            ),
            pre_conv_normalization_config=config.transformer_pre_conv_norm_config,
            feed_forward_block_config=unet_config.FeedForwardBlock2DConfig(
                dim=mid_block_channels,
                hidden_dim=mid_block_channels * 4,
                normalization_config=config.transformer_norm_config,
                activation_config=layers_config.ActivationConfig(
                    type=config.transformer_ff_activation_type,
                    dim_in=mid_block_channels,
                    dim_out=mid_block_channels * 4,
                ),
                use_bias=True,
            ),
        ),
    )
    self._map_mid_block(
        state,
        converted_state,
        self._names.mid_block_tensor_names,
        "mid_block",
        mid_block_config,
    )

    # Map up_decoders.
    reversed_block_out_channels = list(
        reversed(model.config.block_out_channels)
    )
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
        up_encoder_block_config = unet_config.SkipUpDecoderBlock2DConfig(
            in_channels=input_channel,
            out_channels=output_channel,
            prev_out_channels=prev_out_channel,
            normalization_config=config.residual_norm_config,
            activation_config=layers_config.ActivationConfig(
                config.residual_activation_type
            ),
            num_layers=up_decoder_layers_per_block,
            time_embedding_channels=config.time_embedding_blocks_dim,
            add_upsample=not_final_block,
            upsample_conv=True,
            sampling_config=unet_config.UpSamplingConfig(
                mode=unet_config.SamplingType.NEAREST,
                scale_factor=2,
            ),
            transformer_block_config=unet_config.TransformerBlock2DConfig(
                attention_block_config=unet_config.AttentionBlock2DConfig(
                    dim=output_channel,
                    normalization_config=config.transformer_norm_config,
                    attention_config=build_attention_config(
                        num_heads=config.transformer_num_attention_heads,
                        dim=output_channel,
                        num_query_groups=config.transformer_num_attention_heads,
                    ),
                ),
                cross_attention_block_config=unet_config.CrossAttentionBlock2DConfig(
                    query_dim=output_channel,
                    cross_dim=config.transformer_cross_attention_dim,
                    hidden_dim=output_channel,
                    output_dim=output_channel,
                    normalization_config=config.transformer_norm_config,
                    attention_config=build_attention_config(
                        num_heads=config.transformer_num_attention_heads,
                        dim=output_channel,
                        num_query_groups=config.transformer_num_attention_heads,
                    ),
                ),
                pre_conv_normalization_config=config.transformer_pre_conv_norm_config,
                feed_forward_block_config=unet_config.FeedForwardBlock2DConfig(
                    dim=output_channel,
                    hidden_dim=output_channel * 4,
                    normalization_config=config.transformer_norm_config,
                    activation_config=layers_config.ActivationConfig(
                        type=config.transformer_ff_activation_type,
                        dim_in=output_channel,
                        dim_out=output_channel * 4,
                    ),
                    use_bias=True,
                ),
            ),
        )
      else:
        up_encoder_block_config = unet_config.SkipUpDecoderBlock2DConfig(
            in_channels=input_channel,
            out_channels=output_channel,
            prev_out_channels=prev_out_channel,
            normalization_config=config.residual_norm_config,
            activation_config=layers_config.ActivationConfig(
                config.residual_activation_type
            ),
            num_layers=up_decoder_layers_per_block,
            time_embedding_channels=config.time_embedding_blocks_dim,
            add_upsample=not_final_block,
            upsample_conv=True,
            sampling_config=unet_config.UpSamplingConfig(
                mode=unet_config.SamplingType.NEAREST, scale_factor=2
            ),
        )
      self._map_skip_up_decoder_block(
          state,
          converted_state,
          f"up_decoders.{i}",
          up_encoder_block_config,
          self._names.up_decoder_blocks_tensor_names[i],
      )
    if strict and state:
      raise ValueError(
          f"Failed to map all tensor. Remaing tensor are: {list(state.keys())}"
      )
    return model.load_state_dict(converted_state, strict=strict)

  def _map_time_embedding(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      converted_state_param_prefix: str,
      tensor_names: TimeEmbeddingTensorNames,
  ):
    _map_to_converted_state(
        state,
        tensor_names.w1,
        converted_state,
        f"{converted_state_param_prefix}.w1",
    )
    _map_to_converted_state(
        state,
        tensor_names.w2,
        converted_state,
        f"{converted_state_param_prefix}.w2",
    )
