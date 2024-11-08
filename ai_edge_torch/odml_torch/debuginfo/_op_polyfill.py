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
"""Polyfill op for torch.ops.xla.write_mlir_debuginfo.

In odml-torch, MLIR debuginfo is generated in the lowering framework directly
without the need of an additional torch op to write. This file register a no-op
placeholder torch op to replace torch.ops.xla.write_mlir_debuginfo in
ai-edge-torch.
"""

from jax._src.lib.mlir import ir
import torch

from .. import _torch_library
from .. import lowerings


_torch_library.ODML_TORCH_LIB.define(
    "write_mlir_debuginfo(Tensor x, str data) -> Tensor"
)

write_mlir_debuginfo_op = torch.ops.odml_torch.write_mlir_debuginfo


@torch.library.impl(
    _torch_library.ODML_TORCH_LIB,
    "write_mlir_debuginfo",
    "CompositeExplicitAutograd",
)
def write_mlir_debuginfo(x: torch.Tensor, _: str):
  return x


@torch.library.impl(
    _torch_library.ODML_TORCH_LIB, "write_mlir_debuginfo", "Meta"
)
def write_mlir_debuginfo_meta(x: torch.Tensor, _: str):
  return torch.empty_like(x)


@lowerings.lower(torch.ops.odml_torch.write_mlir_debuginfo)
def write_mlir_debuginfo_lowering(lctx, x: ir.Value, _: str):
  return x
