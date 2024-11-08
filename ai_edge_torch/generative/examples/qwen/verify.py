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

"""Verifies the reauthored Qwen 2.5 0.5B, 1.5B, and 3B models."""

import logging
import pathlib

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.qwen import qwen
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
import transformers


_MODEL_SIZE = flags.DEFINE_enum(
    "model_size",
    "3b",
    ["0.5b", "1.5b", "3b"],
    "The size of the model to verify.",
)
_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "What is the meaning of life?",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)

_CHECKPOINT = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}

_BUILDER = {
    "0.5b": qwen.build_0_5b_model,
    "1.5b": qwen.build_1_5b_model,
    "3b": qwen.build_3b_model,
}


def main(_):
  checkpoint = _CHECKPOINT[_MODEL_SIZE.value]
  logging.info("Loading the original model from: %s", checkpoint)
  original_model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint)

  # Locate the cached dir.
  cached_config_file = transformers.utils.cached_file(
      checkpoint, transformers.utils.CONFIG_NAME
  )
  reauthored_checkpoint = pathlib.Path(cached_config_file).parent
  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = _BUILDER[_MODEL_SIZE.value](reauthored_checkpoint)

  logging.info("Loading the tokenizer from: %s", checkpoint)
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

  verifier.verify_reauthored_model(
      original_model=transformers_verifier.TransformersModelWrapper(
          original_model
      ),
      reauthored_model=verifier.ReauthoredModelWrapper(reauthored_model),
      tokenizer=verifier.TokenizerWrapper(tokenizer),
      generate_prompts=_PROMPTS.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
      atol=1e-04,
  )


if __name__ == "__main__":
  app.run(main)
