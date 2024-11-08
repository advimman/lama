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
import glob
import os
from typing import Callable, Dict, List, Tuple

from ai_edge_torch.generative.layers import model_config
from safetensors import safe_open
import torch


def load_safetensors(full_path: str):
  """Loads safetensors into a single state dictionary.

  Args:
    full_path (string): the directory that contains the safetensor files.

  Returns:
    A state dictionary contating loaded tensors.

  Raises:
    ValueError: If no tensors are loaded from the provided directory or file.
  """
  pattern = (
      os.path.join(full_path, "*.safetensors")
      if os.path.isdir(full_path)
      else full_path
  )
  files = []
  for file in glob.glob(pattern):
    files.append(file)

  tensors = {}
  for file in files:
    with safe_open(file, framework="pt") as fp:
      for k in fp.keys():
        assert k not in tensors
        tensors[k] = fp.get_tensor(k)

  if not tensors:
    raise ValueError("Failed to load SafeTensors.")
  return tensors


def load_pytorch_statedict(full_path: str):
  """Loads state dictionary binaries into a single state dictionary.

  Args:
    full_path (string): the directory that contains the bin files.

  Returns:
    A state dictionary contating loaded tensors.

  Raises:
    ValueError: If no tensors are loaded from the provided directory or file.
  """
  files = []
  patterns = []
  if os.path.isdir(full_path):
    patterns.append(os.path.join(full_path, "*.bin"))
    patterns.append(os.path.join(full_path, "*pt"))
  else:
    patterns.append(full_path)
  for pattern in patterns:
    for file in glob.glob(pattern):
      files.append(file)

  tensors = {}
  for file in files:
    this_file_tensors = torch.load(file)
    for k in this_file_tensors:
      assert k not in tensors
    tensors.update(this_file_tensors)

  if not tensors:
    raise ValueError("Failed to load torch bin files.")
  return tensors


class ModelLoader:
  """Utlity for loading model checkpoints to the Edge Generative API layer."""

  @dataclass
  class TensorNames:
    attn_query_proj: str = None
    attn_key_proj: str = None
    attn_value_proj: str = None
    attn_fused_qkv_proj: str = None
    attn_output_proj: str = None
    attn_query_norm: str = None
    attn_key_norm: str = None

    ff_up_proj: str = None
    ff_down_proj: str = None
    ff_gate_proj: str = None

    pre_attn_norm: str = None
    post_attn_norm: str = None
    pre_ff_norm: str = None
    post_ff_norm: str = None
    embedding: str = None
    embedding_position: str = None
    final_norm: str = None
    lm_head: str = None

  def __init__(self, file_name: str, names: TensorNames) -> None:
    """ModelLoader constructor.

    Can be used to load multiple models of the same type.

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
    state = state["model_state_dict"] if "model_state_dict" in state else state
    converted_state = dict()
    if self._names.embedding is not None:
      converted_state["tok_embedding.weight"] = state.pop(
          f"{self._names.embedding}.weight"
      )
      if self._names.embedding_position is not None:
        converted_state["tok_embedding_position"] = state.pop(
            f"{self._names.embedding_position}"
        )
    if self._names.lm_head is not None:
      converted_state["lm_head.weight"] = state.pop(
          f"{self._names.lm_head}.weight"
      )
      if model.config.lm_head_use_bias:
        converted_state["lm_head.bias"] = state.pop(
            f"{self._names.lm_head}.bias"
        )
    if self._names.final_norm is not None:
      final_norm_name = self._names.final_norm
      converted_state["final_norm.weight"] = state.pop(
          f"{final_norm_name}.weight"
      )
      if f"{final_norm_name}.bias" in state:
        converted_state["final_norm.bias"] = state.pop(
            f"{final_norm_name}.bias"
        )

    for i in range(model.config.num_layers):
      self._map_norm(i, model.config, state, converted_state)
      self._map_feedforward(i, model.config, state, converted_state)
      self._map_attention(i, model.config, state, converted_state)

    if strict and state:
      raise ValueError(
          f"Failed to map all tensor. Remaing tensor are: {list(state.keys())}"
      )
    return model.load_state_dict(converted_state, strict=strict)

  def _get_loader(self) -> Callable[[str], Dict[str, torch.Tensor]]:
    """A best effort method for finding appropriate state loader.

    Raises:
        ValueError: If it fails to find an appropriate loader.

    Returns:
        Callable[[str], Dict[str, torch.Tensor]]: State loader to be used.
    """
    if os.path.isdir(self._file_name):
      if glob.glob(os.path.join(self._file_name, "*.safetensors")):
        return load_safetensors
      if glob.glob(os.path.join(self._file_name, "*.bin")) or glob.glob(
          os.path.join(self._file_name, "*pt")
      ):
        return load_pytorch_statedict

    if self._file_name.endswith(".safetensors"):
      return load_safetensors

    if self._file_name.endswith(".bin") or self._file_name.endswith("pt"):
      return load_pytorch_statedict

    raise ValueError("File format not supported.")

  def _map_feedforward(
      self,
      idx: int,
      config: model_config.ModelConfig,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
  ):
    prefix = f"transformer_blocks.{idx}"
    ff_config = config.block_config(idx).ff_config
    if ff_config.type == model_config.FeedForwardType.SEQUENTIAL:
      ff_up_proj_name = self._names.ff_up_proj.format(idx)
      ff_down_proj_name = self._names.ff_down_proj.format(idx)
      converted_state[f"{prefix}.ff.w1.weight"] = state.pop(
          f"{ff_up_proj_name}.weight"
      )
      converted_state[f"{prefix}.ff.w2.weight"] = state.pop(
          f"{ff_down_proj_name}.weight"
      )
      if ff_config.use_bias:
        converted_state[f"{prefix}.ff.w1.bias"] = state.pop(
            f"{ff_up_proj_name}.bias"
        )
        converted_state[f"{prefix}.ff.w2.bias"] = state.pop(
            f"{ff_down_proj_name}.bias"
        )
    else:
      ff_up_proj_name = self._names.ff_up_proj.format(idx)
      ff_down_proj_name = self._names.ff_down_proj.format(idx)
      ff_gate_proj_name = self._names.ff_gate_proj.format(idx)
      converted_state[f"{prefix}.ff.w3.weight"] = state.pop(
          f"{ff_up_proj_name}.weight"
      )
      converted_state[f"{prefix}.ff.w2.weight"] = state.pop(
          f"{ff_down_proj_name}.weight"
      )
      converted_state[f"{prefix}.ff.w1.weight"] = state.pop(
          f"{ff_gate_proj_name}.weight"
      )
      if ff_config.use_bias:
        converted_state[f"{prefix}.ff.w3.bias"] = state.pop(
            f"{ff_up_proj_name}.bias"
        )
        converted_state[f"{prefix}.ff.w2.bias"] = state.pop(
            f"{ff_down_proj_name}.bias"
        )
        converted_state[f"{prefix}.ff.w1.bias"] = state.pop(
            f"{ff_gate_proj_name}.bias"
        )

    if self._names.pre_ff_norm is not None:
      pre_ff_norm_name = self._names.pre_ff_norm.format(idx)
      converted_state[f"{prefix}.ff.pre_ff_norm.weight"] = state.pop(
          f"{pre_ff_norm_name}.weight"
      )
      if f"{pre_ff_norm_name}.bias" in state:
        converted_state[f"{prefix}.ff.pre_ff_norm.bias"] = state.pop(
            f"{pre_ff_norm_name}.bias"
        )

    if self._names.post_ff_norm is not None:
      post_ff_norm_name = self._names.post_ff_norm.format(idx)
      converted_state[f"{prefix}.ff.post_ff_norm.weight"] = state.pop(
          f"{post_ff_norm_name}.weight"
      )
      if f"{post_ff_norm_name}.bias" in state:
        converted_state[f"{prefix}.ff.post_ff_norm.bias"] = state.pop(
            f"{post_ff_norm_name}.bias"
        )

  def _map_attention(
      self,
      idx: int,
      config: model_config.ModelConfig,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
  ):
    prefix = f"transformer_blocks.{idx}"
    attn_config = config.block_config(idx).attn_config
    if self._names.attn_fused_qkv_proj:
      fused_qkv_name = self._names.attn_fused_qkv_proj.format(idx)
      converted_state[f"{prefix}.atten_func.qkv_projection.weight"] = state.pop(
          f"{fused_qkv_name}.weight"
      )
    else:
      q_name = self._names.attn_query_proj.format(idx)
      k_name = self._names.attn_key_proj.format(idx)
      v_name = self._names.attn_value_proj.format(idx)
      converted_state[f"{prefix}.atten_func.qkv_projection.weight"] = (
          self._fuse_qkv(
              attn_config,
              state.pop(f"{q_name}.weight"),
              state.pop(f"{k_name}.weight"),
              state.pop(f"{v_name}.weight"),
          )
      )
    if attn_config.qkv_use_bias:
      if self._names.attn_fused_qkv_proj:
        converted_state[f"{prefix}.atten_func.qkv_projection.bias"] = state.pop(
            f"{fused_qkv_name}.bias"
        )
      else:
        converted_state[f"{prefix}.atten_func.qkv_projection.bias"] = (
            self._fuse_qkv(
                attn_config,
                state.pop(f"{q_name}.bias"),
                state.pop(f"{k_name}.bias"),
                state.pop(f"{v_name}.bias"),
            )
        )

    if self._names.attn_query_norm is not None:
      attn_query_norm_name = self._names.attn_query_norm.format(idx)
      converted_state[f"{prefix}.atten_func.query_norm.weight"] = state.pop(
          f"{attn_query_norm_name}.weight"
      )
    if self._names.attn_key_norm is not None:
      attn_key_norm_name = self._names.attn_key_norm.format(idx)
      converted_state[f"{prefix}.atten_func.key_norm.weight"] = state.pop(
          f"{attn_key_norm_name}.weight"
      )

    o_name = self._names.attn_output_proj.format(idx)
    converted_state[f"{prefix}.atten_func.output_projection.weight"] = (
        state.pop(f"{o_name}.weight")
    )
    if attn_config.output_proj_use_bias:
      converted_state[f"{prefix}.atten_func.output_projection.bias"] = (
          state.pop(f"{o_name}.bias")
      )

  def _map_norm(
      self,
      idx: int,
      config: model_config.ModelConfig,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
  ):
    prefix = f"transformer_blocks.{idx}"
    if self._names.pre_attn_norm is not None:
      pre_attn_norm_name = self._names.pre_attn_norm.format(idx)
      converted_state[f"{prefix}.pre_atten_norm.weight"] = state.pop(
          f"{pre_attn_norm_name}.weight"
      )
      if f"{pre_attn_norm_name}.bias" in state:
        converted_state[f"{prefix}.pre_atten_norm.bias"] = state.pop(
            f"{pre_attn_norm_name}.bias"
        )

    if self._names.post_attn_norm is not None:
      post_attn_norm_name = self._names.post_attn_norm.format(idx)
      converted_state[f"{prefix}.post_atten_norm.weight"] = state.pop(
          f"{post_attn_norm_name}.weight"
      )
      if f"{post_attn_norm_name}.bias" in state:
        converted_state[f"{prefix}.post_atten_norm.bias"] = state.pop(
            f"{post_attn_norm_name}.bias"
        )

  def _fuse_qkv(
      self,
      attn_config: model_config.AttentionConfig,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
  ) -> torch.Tensor:
    if attn_config.qkv_fused_interleaved:
      q_per_kv = attn_config.num_heads // attn_config.num_query_groups
      qs = torch.split(q, attn_config.head_dim * q_per_kv)
      ks = torch.split(k, attn_config.head_dim)
      vs = torch.split(v, attn_config.head_dim)
      cycled = [t for group in zip(qs, ks, vs) for t in group]
      return torch.cat(cycled)
    else:
      return torch.cat([q, k, v], dim=0)
