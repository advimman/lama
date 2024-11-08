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

import argparse
import os
import pathlib
from typing import Optional

import ai_edge_torch
from ai_edge_torch.generative.examples.stable_diffusion import samplers
from ai_edge_torch.generative.examples.stable_diffusion import tokenizer
from ai_edge_torch.generative.examples.stable_diffusion import util
import numpy as np
from PIL import Image
import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--tokenizer_vocab_dir',
    type=str,
    help=(
        'Directory to the tokenizer vocabulary files, which include'
        ' `merges.txt` and `vocab.json`'
    ),
    required=True,
)
arg_parser.add_argument(
    '--clip_ckpt',
    type=str,
    help='Path to CLIP TFLite tflite file',
    required=True,
)
arg_parser.add_argument(
    '--diffusion_ckpt',
    type=str,
    help='Path to diffusion tflite file',
    required=True,
)
arg_parser.add_argument(
    '--decoder_ckpt',
    type=str,
    help='Path to decoder tflite file',
    required=True,
)
arg_parser.add_argument(
    '--output_path',
    type=str,
    help='Path to the output generated image file.',
    required=True,
)
arg_parser.add_argument(
    '--prompt',
    default='a photograph of an astronaut riding a horse',
    type=str,
    help='The prompt to guide the image generation.',
)
arg_parser.add_argument(
    '--n_inference_steps',
    default=20,
    type=int,
    help='The number of denoising steps.',
)
arg_parser.add_argument(
    '--sampler',
    default='k_euler',
    type=str,
    choices=['k_euler', 'k_euler_ancestral', 'k_lms'],
    help=(
        'A sampler to be used to denoise the encoded image latents. Can be one'
        ' of `k_lms, `k_euler`, or `k_euler_ancestral`.'
    ),
)
arg_parser.add_argument(
    '--seed',
    default=None,
    type=int,
    help=(
        'A seed to make generation deterministic. A random number is used if'
        ' unspecified.'
    ),
)


class StableDiffusion:

  def __init__(
      self,
      *,
      tokenizer_vocab_dir: str,
      clip_ckpt: str,
      encoder_ckpt: Optional[str] = None,
      diffusion_ckpt: str,
      decoder_ckpt: str
  ):
    self.tokenizer = tokenizer.Tokenizer(tokenizer_vocab_dir)
    self.clip = ai_edge_torch.model.TfLiteModel.load(clip_ckpt)
    self.decoder = ai_edge_torch.model.TfLiteModel.load(decoder_ckpt)
    self.diffusion = ai_edge_torch.model.TfLiteModel.load(diffusion_ckpt)
    if encoder_ckpt is not None:
      self.encoder = ai_edge_torch.model.TfLiteModel.load(encoder_ckpt)


def run_tflite_pipeline(
    model: StableDiffusion,
    prompt: str,
    output_path: str,
    uncond_prompt: Optional[str] = None,
    cfg_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    sampler: str = 'k_euler',
    n_inference_steps: int = 20,
    seed: Optional[int] = None,
    strength: float = 0.8,
    input_image: Optional[Image.Image] = None,
):
  """Run stable diffusion pipeline with tflite model.

  Args:
    model: StableDiffsuion model.
    prompt: The prompt to guide the image generation.
    output_path: The path to the generated output image.
    uncond_prompt: The prompt not to guide the image generation.
    cfg_scale: Guidance scale of classifier-free guidance. Higher guidance scale
      encourages to generate images that are closely linked to the text
      `prompt`, usually at the expense of lower image quality.
    height: The height in pixels of the generated image.
    width: The width in pixels of the generated image.
    sampler: A sampler to be used to denoise the encoded image latents. Can be
      one of `k_lms, `k_euler`, or `k_euler_ancestral`.
    n_inference_steps: The number of denoising steps. More denoising steps
      usually lead to a higher quality image at the expense of slower inference.
      This parameter will be modulated by `strength`.
    seed: A seed to make generation deterministic.
    strength: Conceptually, indicates how much to transform the reference
      `input_image`. Must be between 0 and 1. `input_image` will be used as a
      starting point, adding more noise to it the larger the `strength`. The
      number of denoising steps depends on the amount of noise initially added.
      When `strength` is 1, added noise will be maximum and the denoising
      process will run for the full number of iterations specified in
      `n_inference_steps`. A value of 1, therefore, essentially ignores
      `input_image`.
    input_image: Image which is served as the starting point for the image
      generation.
  """
  if not 0 < strength < 1:
    raise ValueError('strength must be between 0 and 1')
  if height % 8 or width % 8:
    raise ValueError('height and width must be a multiple of 8')
  if seed is not None:
    np.random.seed(seed)
  if uncond_prompt is None:
    uncond_prompt = ''

  if sampler == 'k_lms':
    sampler = samplers.KLMSSampler(n_inference_steps=n_inference_steps)
  elif sampler == 'k_euler':
    sampler = samplers.KEulerSampler(n_inference_steps=n_inference_steps)
  elif sampler == 'k_euler_ancestral':
    sampler = samplers.KEulerAncestralSampler(
        n_inference_steps=n_inference_steps
    )
  else:
    raise ValueError(
        'Unknown sampler value %s. '
        'Accepted values are {k_lms, k_euler, k_euler_ancestral}' % sampler
    )

  # Text embedding.
  cond_tokens = model.tokenizer.encode(prompt)
  cond_context = model.clip(
      np.array(cond_tokens).astype(np.int32), signature_name='encode'
  )
  uncond_tokens = model.tokenizer.encode(uncond_prompt)
  uncond_context = model.clip(
      np.array(uncond_tokens).astype(np.int32), signature_name='encode'
  )
  context = np.concatenate([cond_context, uncond_context], axis=0)
  noise_shape = (1, 4, height // 8, width // 8)

  # Initialization starts from input_image if any, otherwise, starts from a
  # random sampling.
  if input_image:
    if not hasattr(model, 'encoder'):
      raise AttributeError(
          'Stable Diffusion must be initialized with encoder to accept'
          ' input_image.'
      )
    input_image = input_image.resize((width, height))
    input_image_np = util.rescale(input_image, (0, 255), (-1, 1))
    input_image_np = util.move_channel(input_image_np, to='first')
    encoder_noise = np.random.normal(size=noise_shape).astype(np.float32)
    latents = model.encoder(input_image_np.astype(np.float32), encoder_noise)
    latents_noise = np.random.normal(size=noise_shape).astype(np.float32)
    sampler.set_strength(strength=strength)
    latents += latents_noise * sampler.initial_scale
  else:
    latents = np.random.normal(size=noise_shape).astype(np.float32)
    latents *= sampler.initial_scale

  # Diffusion process.
  timesteps = tqdm.tqdm(sampler.timesteps)
  for _, timestep in enumerate(timesteps):
    time_embedding = util.get_time_embedding(timestep)

    input_latents = latents * sampler.get_input_scale()
    input_latents = input_latents.repeat(2, axis=0)
    output = model.diffusion(
        input_latents.astype(np.float32),
        context.astype(np.float32),
        time_embedding,
        signature_name='diffusion',
    )
    output_cond, output_uncond = np.split(output, 2, axis=0)
    output = cfg_scale * (output_cond - output_uncond) + output_uncond

    latents = sampler.step(latents, output)

  # Image decoding.
  images = model.decoder(latents.astype(np.float32), signature_name='decode')
  images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
  images = util.move_channel(images, to='last')
  if not os.path.exists(output_path):
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(images[0].astype(np.uint8)).save(output_path)


if __name__ == '__main__':
  args = arg_parser.parse_args()
  run_tflite_pipeline(
      StableDiffusion(
          tokenizer_vocab_dir=args.tokenizer_vocab_dir,
          clip_ckpt=args.clip_ckpt,
          diffusion_ckpt=args.diffusion_ckpt,
          decoder_ckpt=args.decoder_ckpt,
      ),
      prompt=args.prompt,
      output_path=args.output_path,
      sampler=args.sampler,
      n_inference_steps=args.n_inference_steps,
      seed=args.seed,
  )
