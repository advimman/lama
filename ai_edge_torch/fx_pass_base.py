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

import abc
import collections
from typing import Sequence, Union

import torch
from torch.fx.passes.infra.pass_base import PassBase
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import pass_result_wrapper
import torch.utils._pytree as pytree

FxPassBase = PassBase
FxPassResult = PassResult
ExportedProgramPassResult = collections.namedtuple(
    "ExportedProgramPassResult", ["exported_program", "modified"]
)


class ExportedProgramPassBase(abc.ABC):

  def __call__(
      self, exported_program: torch.export.ExportedProgram
  ) -> ExportedProgramPassResult:
    self.requires(exported_program)
    res = self.call(exported_program)
    self.ensures(exported_program)
    return res

  @abc.abstractmethod
  def call(
      self, exported_program: torch.export.ExportedProgram
  ) -> ExportedProgramPassResult:
    pass

  def requires(self, exported_program: torch.export.ExportedProgram) -> None:
    pass

  def ensures(self, exported_program: torch.export.ExportedProgram) -> None:
    pass


# TODO(cnchan): make a PassManager class.
def run_passes(
    exported_program: torch.export.ExportedProgram,
    passes: Sequence[Union[ExportedProgramPassBase, FxPassBase]],
) -> torch.export.ExportedProgram:
  passes, _ = pytree.tree_flatten(passes)
  for pass_ in passes:
    if not isinstance(pass_, ExportedProgramPassBase):
      pass_ = pass_result_wrapper(pass_)
    if isinstance(pass_, ExportedProgramPassBase):
      exported_program = pass_(exported_program).exported_program
    else:
      gm = exported_program.graph_module
      gm, modified = pass_(gm)
      if modified and gm is not exported_program.graph_module:
        exported_program = torch.export.ExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=exported_program.graph_signature,
            state_dict=exported_program.state_dict,
            range_constraints=exported_program.range_constraints,
            module_call_graph=exported_program.module_call_graph,
            example_inputs=exported_program.example_inputs,
            verifiers=exported_program.verifiers,
            constants=exported_program.constants,
        )
  return exported_program


class CanonicalizePass(ExportedProgramPassBase):

  # A dummy decomp table for running ExportedProgram.run_decompositions without
  # any op decompositions but just aot_export_module. Due to the check in
  # run_decompositions, if None or an empty dict is passed as decomp_table,
  # it will run the default aten-coreaten decompositions. Therefore a non-empty
  # dummy decomp table is needed.
  # Ref: https://github.com/pytorch/pytorch/blob/db895ace1d36726e64781774f53b3d3098206116/torch/export/exported_program.py#L543
  _DUMMY_DECOMP_TABLE = {
      torch._ops.OperatorBase(): lambda: None,
  }

  def call(self, exported_program: torch.export.ExportedProgram):
    for node in exported_program.graph.nodes:
      if node.target == torch.ops.aten.view.default:
        # Passes or torch.export may generate aten.view nodes not respecting the
        # tensor memory format. Changes all the aten.view to torch.reshape
        # for retracing. If the input memory format is already contiguous,
        # retracing in run_decomposition below would decompose torch.reshape
        # back to one aten.view.
        node.target = lambda self, size: torch.reshape(self.contiguous(), size)

    exported_program = exported_program.run_decompositions(
        self._DUMMY_DECOMP_TABLE
    )
    return ExportedProgramPassResult(exported_program, True)
