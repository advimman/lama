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
import pathlib

from absl import app
from absl import flags
import ai_edge_torch
from ai_edge_torch.generative.examples.stable_diffusion import clip
from ai_edge_torch.generative.examples.stable_diffusion import decoder
from ai_edge_torch.generative.examples.stable_diffusion import diffusion
from ai_edge_torch.generative.examples.stable_diffusion import util
from ai_edge_torch.generative.quantize import quant_recipes
from ai_edge_torch.generative.utilities import stable_diffusion_loader
import torch

_CLIP_CKPT = flags.DEFINE_string(
    'clip_ckpt',
    None,
    help='Path to source CLIP model checkpoint',
    required=True,
)

_DIFFUSION_CKPT = flags.DEFINE_string(
    'diffusion_ckpt',
    None,
    help='Path to source diffusion model checkpoint',
    required=True,
)

_DECODER_CKPT = flags.DEFINE_string(
    'decoder_ckpt',
    None,
    help='Path to source image decoder model checkpoint',
    required=True,
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    help='Path to the converted TF Lite directory.',
    required=True,
)

_QUANTIZE = flags.DEFINE_bool(
    'quantize',
    help='Whether to quantize the model during conversion.',
    default=True,
)

_DEVICE_TYPE = flags.DEFINE_string(
    'device_type',
    'cpu',
    help='The device type of the model. Currently supported: cpu, gpu.',
)


@torch.inference_mode
def convert_stable_diffusion_to_tflite(
    output_dir: str,
    clip_ckpt_path: str,
    diffusion_ckpt_path: str,
    decoder_ckpt_path: str,
    image_height: int = 512,
    image_width: int = 512,
    quantize: bool = True,
):

  clip_model = clip.CLIP(clip.get_model_config())
  loader = stable_diffusion_loader.ClipModelLoader(
      clip_ckpt_path,
      clip.TENSOR_NAMES,
  )
  loader.load(clip_model, strict=False)

  diffusion_model = diffusion.Diffusion(
      diffusion.get_model_config(batch_size=2, device_type=_DEVICE_TYPE.value)
  )
  diffusion_loader = stable_diffusion_loader.DiffusionModelLoader(
      diffusion_ckpt_path, diffusion.TENSOR_NAMES
  )
  diffusion_loader.load(diffusion_model, strict=False)

  decoder_model = decoder.Decoder(
      decoder.get_model_config(device_type=_DEVICE_TYPE.value)
  )
  decoder_loader = stable_diffusion_loader.AutoEncoderModelLoader(
      decoder_ckpt_path, decoder.TENSOR_NAMES
  )
  decoder_loader.load(decoder_model, strict=False)

  # TODO(yichunk): enable image encoder conversion
  # if encoder_ckpt_path is not None:
  #   encoder = Encoder()
  #   encoder.load_state_dict(torch.load(encoder_ckpt_path))

  # Tensors used to trace the model graph during conversion.
  n_tokens = 77
  timestamp = 0
  len_prompt = 1
  prompt_tokens = torch.full((1, n_tokens), 0, dtype=torch.int)
  input_image = torch.full(
      (1, 3, image_height, image_width), 0, dtype=torch.float32
  )
  noise = torch.full(
      (len_prompt, 4, image_height // 8, image_width // 8),
      0,
      dtype=torch.float32,
  )

  input_latents = torch.zeros_like(noise)
  context_cond = clip_model(prompt_tokens)
  context_uncond = torch.zeros_like(context_cond)
  context = torch.cat([context_cond, context_uncond], axis=0)
  time_embedding = util.get_time_embedding(timestamp)

  if not os.path.exists(output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

  quant_config = (
      quant_recipes.full_int8_weight_only_recipe() if quantize else None
  )

  # TODO(yichunk): convert to multi signature tflite model.
  # CLIP text encoder
  ai_edge_torch.signature('encode', clip_model, (prompt_tokens,)).convert(
      quant_config=quant_config
  ).export(f'{output_dir}/clip.tflite')

  # TODO(yichunk): enable image encoder conversion
  # Image encoder
  # ai_edge_torch.signature('encode', encoder, (input_image, noise)).convert(quant_config=quant_config).export(
  #     f'{output_dir}/encoder.tflite'
  # )

  # Diffusion
  ai_edge_torch.signature(
      'diffusion',
      diffusion_model,
      (torch.repeat_interleave(input_latents, 2, 0), context, time_embedding),
  ).convert(quant_config=quant_config).export(f'{output_dir}/diffusion.tflite')

  # Image decoder
  ai_edge_torch.signature('decode', decoder_model, (input_latents,)).convert(
      quant_config=quant_config
  ).export(f'{output_dir}/decoder.tflite')


def main(_):
  convert_stable_diffusion_to_tflite(
      output_dir=_OUTPUT_DIR.value,
      clip_ckpt_path=_CLIP_CKPT.value,
      diffusion_ckpt_path=_DIFFUSION_CKPT.value,
      decoder_ckpt_path=_DECODER_CKPT.value,
      quantize=_QUANTIZE.value,
  )


if __name__ == '__main__':
  app.run(main)
