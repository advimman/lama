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

from ai_edge_torch.generative.examples.stable_diffusion import util
from ai_edge_torch.generative.examples.stable_diffusion.samplers.sampler import SamplerInterface  # NOQA
import numpy as np


class KEulerAncestralSampler(SamplerInterface):

  def __init__(self, n_inference_steps=50, n_training_steps=1000):
    timesteps = np.linspace(n_training_steps - 1, 0, n_inference_steps)

    alphas_cumprod = util.get_alphas_cumprod(n_training_steps=n_training_steps)
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    log_sigmas = np.log(sigmas)
    log_sigmas = np.interp(timesteps, range(n_training_steps), log_sigmas)
    sigmas = np.exp(log_sigmas)
    sigmas = np.append(sigmas, 0)

    self.sigmas = sigmas
    self.initial_scale = sigmas.max()
    self.timesteps = timesteps
    self.n_inference_steps = n_inference_steps
    self.n_training_steps = n_training_steps
    self.step_count = 0

  def get_input_scale(self, step_count=None):
    if step_count is None:
      step_count = self.step_count
    sigma = self.sigmas[step_count]
    return 1 / (sigma**2 + 1) ** 0.5

  def set_strength(self, strength=1):
    start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
    self.timesteps = np.linspace(
        self.n_training_steps - 1, 0, self.n_inference_steps
    )
    self.timesteps = self.timesteps[start_step:]
    self.initial_scale = self.sigmas[start_step]
    self.step_count = start_step

  def step(self, latents, output):
    t = self.step_count
    self.step_count += 1

    sigma_from = self.sigmas[t]
    sigma_to = self.sigmas[t + 1]
    sigma_up = sigma_to * (1 - (sigma_to**2 / sigma_from**2)) ** 0.5
    sigma_down = sigma_to**2 / sigma_from
    latents += output * (sigma_down - sigma_from)
    noise = np.random.normal(size=latents.shape)
    latents += noise * sigma_up
    return latents
