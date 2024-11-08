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

import re
from typing import Optional
from ai_edge_torch import config
from absl.testing import absltest as googletest


def _extract_backend_configs(mlir):
  mlir = mlir.replace("\\22", '"')
  configs = []
  for match in re.finditer(r"backend_config\s*=\s*\"(\{.*\})\"", mlir):
    configs.append(match.group(1))
  return "\n".join(configs)


def assert_string_count(
    test_case: googletest.TestCase,
    mlir: str,
    torch_xla_pattern_counter: dict[str, int],
    odml_torch_pattern_counter: dict[str, int],
    odml_torch_attr_counter: Optional[dict[str, int]] = None,
):

  if odml_torch_attr_counter is None:
    odml_torch_attr_counter = {}

  if config.Config.use_torch_xla:
    for key in torch_xla_pattern_counter:
      test_case.assertEqual(
          mlir.count(key),
          torch_xla_pattern_counter[key],
      )
  else:
    for key in odml_torch_pattern_counter:
      test_case.assertEqual(
          mlir.count(key),
          odml_torch_pattern_counter[key],
      )
    backend_configs = _extract_backend_configs(mlir)
    print("backend_configs:")
    print(backend_configs)
    for key in odml_torch_attr_counter:
      test_case.assertEqual(
          backend_configs.count(key),
          odml_torch_attr_counter[key],
      )
