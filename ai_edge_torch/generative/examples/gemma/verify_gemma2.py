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

"""Verifies the reauthored Gemma2 model."""

import logging
from absl import app
from absl import flags
from ai_edge_torch.generative.examples.gemma import gemma2
from ai_edge_torch.generative.examples.gemma import verify_util
import kagglehub


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


def main(_):
  checkpoint = kagglehub.model_download("google/gemma-2/pyTorch/gemma-2-2b-it")

  logging.info("Building the reauthored model from: %s", checkpoint)
  reauthored_model = gemma2.build_2b_model(checkpoint)

  verify_util.verify_reauthored_gemma_model(
      checkpoint=checkpoint,
      variant="2b-v2",
      reauthored_model=reauthored_model,
      generate_prompts=_PROMPTS.value,
      forward_input_ids=[[2, 651, 9456, 576, 573, 3520, 3858, 603, 235248]],
      max_new_tokens=_MAX_NEW_TOKENS.value,
      atol=1e-04,
  )


if __name__ == "__main__":
  app.run(main)
