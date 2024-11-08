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

"""Verifies the reauthored Phi-2 model."""
import logging

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.phi import phi2
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
import kagglehub
import transformers


_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "Instruct: Write an email about the weather Output:",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)


def main(_):
  checkpoint = kagglehub.model_download("Microsoft/phi/transformers/2")
  logging.info("Loading the original model from: %s", checkpoint)
  original_model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint)

  logging.info("Building the reauthored model from: %s", checkpoint)
  reauthored_model = phi2.build_model(checkpoint)

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
      atol=1e-03,
  )


if __name__ == "__main__":
  app.run(main)
