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

import abc

import numpy as np


class SamplerInterface(abc.ABC):

  @abc.abstractmethod
  def get_input_scale(self, step_count: int = 1) -> float:
    """Get the input scale of the random samples from sampled distribution"""
    return NotImplemented

  @abc.abstractmethod
  def set_strength(self, strength: float = 1) -> None:
    """Set the strength of initial step.

    Conceptually, indicates how much to transform the reference `input_images`.
    """
    return NotImplemented

  @abc.abstractmethod
  def step(self, latents: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Update latents from the diffusion output by a step"""
    return NotImplemented
