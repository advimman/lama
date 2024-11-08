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

"""Common utils for testing."""

from ai_edge_torch import model
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.lowertools import common_utils
import numpy as np
import torch
from torch.utils import _pytree as pytree


def compare_tflite_torch(
    edge_model: model.Model,
    torch_model: torch.nn.Module,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
    kv_cache: kv_utils.KVCache,
    signature_name: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
  """Compares torch models and TFLite models."""
  values, spec = pytree.tree_flatten({"kv_cache": kv_cache})
  flat_names = common_utils.flat_dict_names(spec.children_specs, spec.context)
  torch_output = torch_model(tokens, input_pos, kv_cache)

  input_kv_flatten = {k: v.numpy() for k, v in zip(flat_names, values)}
  edge_output = edge_model(
      signature_name=signature_name,
      tokens=tokens.numpy(),
      input_pos=input_pos.numpy(),
      **input_kv_flatten,
  )

  return np.allclose(
      edge_output["logits"],
      torch_output["logits"].detach().numpy(),
      atol=atol,
      rtol=rtol,
  )
