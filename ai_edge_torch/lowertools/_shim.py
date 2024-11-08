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

from typing import Any, Optional

from ai_edge_torch import config
from ai_edge_torch._convert import signature
from ai_edge_torch.quantize import quant_config as qcfg
import torch

# isort: off
if config.Config.use_torch_xla:
  from ai_edge_torch.lowertools import torch_xla_utils as utils
  from ai_edge_torch.lowertools.torch_xla_utils import exported_program_to_mlir_text
  from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder
  from torch_xla.experimental.xla_marker import serialize_composite_attr
  # The following imports are needed to register the needed torch_xla ops.
  import torch_xla.experimental.xla_marker
  import torch_xla.experimental.xla_mlir_debuginfo

  mark_tensor_op = torch.ops.xla.mark_tensor.default
  write_mlir_debuginfo_op = torch.ops.xla.write_mlir_debuginfo.default
else:
  from ai_edge_torch.lowertools import odml_torch_utils as utils
  from ai_edge_torch.lowertools.odml_torch_utils import exported_program_to_mlir_text
  from ai_edge_torch.odml_torch.composite import StableHLOCompositeBuilder
  from ai_edge_torch.odml_torch.composite.mark_tensor import serialize_composite_attr
  from ai_edge_torch.odml_torch.composite.mark_tensor import mark_tensor_op
  from ai_edge_torch.odml_torch.debuginfo import write_mlir_debuginfo_op
# isort: on


def exported_programs_to_tflite(
    exported_programs: list[torch.export.ExportedProgram],
    signatures: list[signature.Signature],
    *,
    quant_config: Optional[qcfg.QuantConfig] = None,
    _tfl_converter_flags: Optional[dict[str, Any]] = None,
):
  """Converts a list of ExportedProgram to a TFLite model.

  Args:
    exported_programs: A list of ExportedProgram.
    signatures: A list of Signature.
    quant_config: A QuantConfig.
    _tfl_converter_flags: A dict of flags for TFLiteConverter.

  Returns:
    A TFLite model.
  """
  if _tfl_converter_flags is None:
    _tfl_converter_flags = {}

  bundles: list[utils.MlirBundle] = [
      utils.exported_program_to_mlir(exported, sig.flat_args)
      for exported, sig in zip(exported_programs, signatures)
  ]

  merged_bundle: utils.MergedBundle = utils.merge_mlir_bundles(
      bundles, signatures, exported_programs
  )

  return utils.merged_bundle_to_tfl_model(
      merged_bundle,
      signatures,
      quant_config=quant_config,
      _tfl_converter_flags=_tfl_converter_flags,
  )
