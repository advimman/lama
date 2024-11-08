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
"""Pass to remove all non user outputs from exported program."""


from ai_edge_torch import fx_pass_base
import torch


class RemoveNonUserOutputsPass(fx_pass_base.ExportedProgramPassBase):
  """This pass removes all non user outputs from the exported program's output.

  The FX graph may output more tensors/data than what user's original model
  returns. Those additional outputs include user input mutations, gradient to
  parameter, etc. Those outputs are not supported by our inference only
  conversion or runtime. This pass remove all those outputs to ensure the
  converted models' outputs match what returned from user's model in eval mode.
  """

  def call(self, exported_program: torch.export.ExportedProgram):
    for node in exported_program.graph.nodes:
      if node.op != "output":
        continue

      outputs = node.args[0]
      output_specs = exported_program.graph_signature.output_specs

      new_outputs = []
      new_output_specs = []
      for output, spec in zip(outputs, output_specs):
        if spec.kind == torch.export.graph_signature.OutputKind.USER_OUTPUT:
          new_outputs.append(output)
          new_output_specs.append(spec)

      node.args = (tuple(new_outputs),)
      exported_program.graph_signature.output_specs = new_output_specs

    exported_program.graph_module.graph.lint()
    exported_program.graph_module.recompile()
    return fx_pass_base.ExportedProgramPassResult(exported_program, True)
