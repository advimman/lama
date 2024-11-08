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
"""Define context object for export and MLIR lowerings."""

import dataclasses
from jax._src.lib.mlir import ir
import torch


@dataclasses.dataclass
class LoweringContext:
  """The context object used in export interpreter and MLIR lowerings."""

  ir_context: ir.Context
  ir_module: ir.Module
  ir_location: ir.Location = None
  node: torch.fx.Node = None

  @property
  def ctx(self):
    """Shortcut for ir_context."""
    return self.ir_context

  @property
  def loc(self):
    """Shortcut for ir_location."""
    return self.ir_location

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)
