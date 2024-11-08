# Copyright 2024 The AI Edge Torch Authors. All Rights Reserved.
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


def get_supported_layer_schemes():
  """List of layer-scoped quantization schemes supported in runtime.

  Returns:
    List of tuple(activation_dtype, weight_dtype, mode, algorithm, granularity).
  """
  from ai_edge_torch.generative.quantize.quant_attrs import Algorithm as _a
  from ai_edge_torch.generative.quantize.quant_attrs import Dtype as _t
  from ai_edge_torch.generative.quantize.quant_attrs import Granularity as _g
  from ai_edge_torch.generative.quantize.quant_attrs import Mode as _m

  return [
      (_t.FP32, _t.INT8, _m.DYNAMIC_RANGE, _a.MIN_MAX, _g.CHANNELWISE),
      (_t.FP32, _t.INT8, _m.WEIGHT_ONLY, _a.MIN_MAX, _g.CHANNELWISE),
      (_t.FP32, _t.FP16, _m.WEIGHT_ONLY, _a.FLOAT_CAST, _g.NONE),
  ]
