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
import json
from typing import Sequence, Union

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch

from .. import _torch_library
from .. import lowerings

CompositeAttrType = dict[
    str,
    Union[
        int,
        float,
        bool,
        str,
        Sequence[int],
        Sequence[float],
        Sequence[bool],
    ],
]


def _assert_valid_composite_attr(attr: CompositeAttrType):
  if attr is None:
    return
  if not isinstance(attr, dict):
    raise ValueError("Composite attr must be a Python dictionary.")

  for k, v in attr.items():
    if not isinstance(k, str):
      raise ValueError("Composite attr name must be a Python str.")

    invalid_attr_value_error = ValueError(
        "Composite attr value must be either Python str, float, int, bool,"
        " list[int], list[float], list[bool]."
    )
    if isinstance(v, (list, tuple)):
      eltys = {type(el) for el in v}
      if len(eltys) > 1 or next(iter(eltys)) not in (int, float, bool):
        raise invalid_attr_value_error
    elif type(v) not in (str, float, int, bool):
      raise invalid_attr_value_error


@torch._dynamo.assume_constant_result
def serialize_composite_attr(attr: Union[CompositeAttrType, None]):
  """Serialize the composite attr into a dynamo-tracable value."""
  if attr is None:
    return None
  _assert_valid_composite_attr(attr)
  return tuple(attr.items())


@torch._dynamo.assume_constant_result
def deserialize_composite_attr(serialized_attr) -> CompositeAttrType:
  """Deserialize dynamo-tracable composite attribute into its raw value."""
  if serialized_attr is None:
    return None
  return dict(serialized_attr)


_torch_library.ODML_TORCH_LIB.define(
    "mark_tensor(Tensor x, str name, int pos, str id, bool is_input, Any?"
    " attr=None) -> Tensor"
)

mark_tensor_op = torch.ops.odml_torch.mark_tensor.default


@torch.library.impl(
    _torch_library.ODML_TORCH_LIB, "mark_tensor", "CompositeExplicitAutograd"
)
def mark_tensor(
    x: torch.Tensor, name: str, pos: int, id: str, is_input: bool, attr=None
):
  return x


@torch.library.impl(_torch_library.ODML_TORCH_LIB, "mark_tensor", "Meta")
def mark_tensor_meta(
    x: torch.Tensor, name: str, pos: int, id: str, is_input: bool, attr=None
):
  return torch.empty_like(x)


@lowerings.lower(torch.ops.odml_torch.mark_tensor)
def mark_tensor_lowering(
    lctx, x: ir.Value, name: str, pos: int, id: str, is_input: bool, attr=None
):
  attr = deserialize_composite_attr(attr)
  return stablehlo.custom_call(
      [x.type],
      inputs=[x],
      call_target_name="mark_tensor",
      backend_config=ir.StringAttr.get(
          json.dumps({
              "name": name,
              "pos": pos,
              "id": id,
              "is_input": is_input,
              "attr": attr,
          })
      ),
  )
