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

import os

import numpy as np
import torch


def get_time_embedding(timestep):
  freqs = torch.pow(
      10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160
  )
  x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
  return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def get_alphas_cumprod(
    beta_start=0.00085, beta_end=0.0120, n_training_steps=1000
):
  betas = (
      np.linspace(
          beta_start**0.5, beta_end**0.5, n_training_steps, dtype=np.float32
      )
      ** 2
  )
  alphas = 1.0 - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  return alphas_cumprod


def get_file_path(filename, url=None):
  module_location = os.path.dirname(os.path.abspath(__file__))
  parent_location = os.path.dirname(module_location)
  file_location = os.path.join(parent_location, "data", filename)
  return file_location


def move_channel(image, to):
  if to == "first":
    if isinstance(image, torch.Tensor):
      return image.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    if isinstance(image, np.ndarray):
      return image.transpose(0, 3, 1, 2)
  elif to == "last":
    if isinstance(image, torch.Tensor):
      return image.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    if isinstance(image, np.ndarray):
      return image.transpose(0, 2, 3, 1)
  else:
    raise ValueError("to must be one of the following: first, last")


def rescale(x, old_range, new_range, clamp=False):
  old_min, old_max = old_range
  new_min, new_max = new_range
  x -= old_min
  x *= (new_max - new_min) / (old_max - old_min)
  x += new_min
  if clamp:
    if isinstance(x, torch.Tensor):
      x = x.clamp(new_min, new_max)
    elif isinstance(x, np.ndarray):
      x = x.clip(new_min, new_max)
  return x
