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
from ai_edge_torch import odml_torch
import numpy as np
import torch
from torch.utils import _pytree as pytree

from absl.testing import absltest as googletest
from absl.testing import parameterized


def export_without_scalar_inputs(model, args, kwargs):
  export_args = []
  keys = []

  for key, arg in [*enumerate(args), *kwargs.items()]:
    if isinstance(arg, torch.Tensor):
      export_args.append(arg)
      keys.append(key)

  class ModuleWrapper(torch.nn.Module):

    def __init__(self, func, original_args, original_kwargs):
      super().__init__()
      self.original_args = [*original_args]
      self.original_kwargs = original_kwargs.copy()
      self.func = func

    def forward(self, *export_args):
      args = [*self.original_args]
      kwargs = self.original_kwargs.copy()

      for key, arg in zip(keys, export_args):
        if isinstance(key, int):
          args[key] = arg
        else:
          kwargs[key] = arg
      return self.func(*args, **kwargs)

  export_args = tuple(export_args)
  export_kwargs = {}
  return (
      torch.export.export(
          ModuleWrapper(model, args, kwargs).eval(),
          export_args,
          export_kwargs,
      ),
      export_args,
      export_kwargs,
  )


def rnd(dtype, shape, min_v=None, max_v=None):
  """Shortcut for creating a random torch tensor."""
  if dtype in (torch.int32, torch.int64, torch.bool):
    min_v = min_v if min_v else 0
    max_v = max_v if max_v else 10
    return torch.randint(min_v, max_v, shape).to(dtype)
  else:
    min_v = min_v if min_v else 0.0
    max_v = max_v if max_v else 1.0
    return (torch.rand(shape) * (max_v - min_v) + min_v).to(dtype)


class TestCoreAtenOps(parameterized.TestCase):
  """Test core aten ops lowering and validation.

  Source:
  https://github.com/pytorch/xla/blob/master/experimental/torch_xla2/test/test_core_aten_ops.py
  """

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def _diff_output(self, output1, output2, rtol, atol, equal_nan=True):
    """Assert two outputs are numerically equal."""
    if isinstance(output1, (tuple, list)):
      self.assertIsInstance(output2, (tuple, list))

    output1 = pytree.tree_flatten([output1])[0]
    output2 = pytree.tree_flatten([output2])[0]

    self.assertEqual(len(output1), len(output2))
    for o1, o2 in zip(output1, output2):
      o1 = np.array(o1)
      o2 = np.array(o2)
      self.assertEqual(o1.dtype, o2.dtype)
      if not np.allclose(o1, o2, atol=atol, rtol=rtol, equal_nan=equal_nan):
        self.fail('"%r" not close to "%r"' % (o1, o2))

  def _run_export_and_compare(
      self,
      func,
      args,
      kwargs,
      atol=1e-3,
      rtol=1e-5,
      equal_nan=True,
      ignore_indices=False,
  ):
    """Assert func, args, and kwargs can be lowered and pass numerical validation."""
    with self.subTest("torch_eval"):
      res = func(*args, **kwargs)
      with self.subTest("lower"):
        ep, args, kwargs = export_without_scalar_inputs(func, args, kwargs)
        lowered = odml_torch.export.exported_program_to_mlir(ep)

        np_args, np_kwargs = pytree.tree_map_only(
            torch.is_tensor, lambda x: x.detach().numpy(), [args, kwargs]
        )
        res2 = lowered(*np_args, **np_kwargs)
        with self.subTest("torch_lower_eval_diff:" + str(atol)):
          if ignore_indices and isinstance(res, tuple) and len(res) == 2:
            res = res[0]
            res2 = res2[0]

          self._diff_output(
              res,
              res2,
              atol=atol,
              rtol=rtol,
              equal_nan=equal_nan,
          )

  @parameterized.named_parameters(
      # fmt: off
      # pyformat: disable
      ("aten_abs_0", torch.ops.aten.abs, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_acos_0", torch.ops.aten.acos, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_acosh_0", torch.ops.aten.acosh, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_unsqueeze_0", torch.ops.aten.unsqueeze, (rnd(torch.float32, (1, 3, 10)), -2,), dict()),
      ("aten_unsqueeze_3", torch.ops.aten.unsqueeze, (rnd(torch.float32, (1, 3, 10)), -2,), dict()),
      ("aten_unsqueeze_6", torch.ops.aten.unsqueeze, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten__adaptive_avg_pool2d_0", torch.ops.aten._adaptive_avg_pool2d, (rnd(torch.float32, (1, 3, 1, 10)), [1, 5],), dict()),
      ("aten__adaptive_avg_pool2d_1", torch.ops.aten._adaptive_avg_pool2d, (rnd(torch.float32, (1, 3, 10, 10)), [5, 5],), dict()),
      ("aten_squeeze_dim_0", torch.ops.aten.squeeze.dim, (rnd(torch.float32, (1, 3, 1, 5)), -2,), dict()),
      ("aten_squeeze_dim_1", torch.ops.aten.squeeze.dim, (rnd(torch.float32, (1, 3, 1, 5)), -2,), dict()),
      ("aten_squeeze_dim_2", torch.ops.aten.squeeze.dim, (rnd(torch.float32, (10, 10)), 1,), dict()),
      # ("aten__adaptive_avg_pool3d_0", torch.ops.aten._adaptive_avg_pool3d, (rnd(torch.float32, (1, 3, 10, 10, 10)), [5, 5, 5],), dict()),
      # ("aten_add_Scalar_0", torch.ops.aten.add.Scalar, (rnd(torch.float32, (10, 10)), 0.1,), dict()),
      ("aten_add_Tensor_0", torch.ops.aten.add.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_addmm_0", torch.ops.aten.addmm, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_alias_0", torch.ops.aten.alias, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_amax_0", torch.ops.aten.amax, (rnd(torch.float32, (10, 10)), [0, 1],), dict()),
      ("aten_amin_0", torch.ops.aten.amin, (rnd(torch.float32, (10, 10)), [0, 1],), dict()),
      ("aten_any_0", torch.ops.aten.any, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_any_dim_0", torch.ops.aten.any.dim, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten_any_dims_0", torch.ops.aten.any.dim, (rnd(torch.float32, (10, 10)), 0,), dict()),
      ("aten_argmax_0", torch.ops.aten.argmax, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_argmin_0", torch.ops.aten.argmin, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_as_strided_0", torch.ops.aten.as_strided, (rnd(torch.float32, (10, 10)), [2, 2, 2], [8, 4, 1],), dict()),
      ("aten_as_strided_copy_0", torch.ops.aten.as_strided_copy, (rnd(torch.float32, (10, 10)), [5, 5], [2, 2],), dict()),
      ("aten_asin_0", torch.ops.aten.asin, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_asinh_0", torch.ops.aten.asinh, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_atan_0", torch.ops.aten.atan, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_atan2_0", torch.ops.aten.atan2, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_atanh_0", torch.ops.aten.atanh, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_avg_pool2d_0", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 1, 10)), [1, 2], [1, 2],), dict()),
      ("aten_avg_pool2d_1", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (3, 2, 10)), [2, 2], [1, 1], [1, 1],), dict()),
      ("aten_avg_pool2d_stride", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3], [2, 2], [0, 0], False, True, None), dict()),
      ("aten_avg_pool2d_default_values", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3]), dict()),
      ("aten_avg_pool2d_padding", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3], [1, 1], [1, 1], False, True, None), dict()),
      ("aten_avg_pool2d_padding_2", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3], [1, 1], [0, 1], False, True, None), dict()),
      ("aten_avg_pool2d_stride_padding", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3], [2, 2], [1, 1], False, True, None), dict()),
      ("aten_avg_pool2d_no_count_include_pad", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3], [1, 1], [1, 1], False, False, None), dict()),
      ("aten_avg_pool2d_ceil_mode", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3], [1, 1], [1, 1], True, True, None), dict()),
      # ("aten_avg_pool2d_ceil_mode_stride_3", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3], [3, 3], [1, 1], True, True, None), dict()),
      ("aten_avg_pool2d_divisor_override", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3], [1, 1], 0, False, True, 6), dict()),
      ("aten_avg_pool2d_padding_num", torch.ops.aten.avg_pool2d, (rnd(torch.float32, (1, 3, 6, 6)), [3, 3], [1, 1], 1, False, True, None), dict()),
      # ("aten_avg_pool3d_0", torch.ops.aten.avg_pool3d, (rnd(torch.float32, (1, 3, 10, 10, 10)), [2, 2, 2], [2, 2, 2], [0, 0, 0], False, False,), dict()),
      ("aten_bmm_0", torch.ops.aten.bmm, (rnd(torch.float32, (10, 10, 10)), rnd(torch.float32, (10, 10, 10)),), dict()),
      ("aten_cat_0", torch.ops.aten.cat, ([torch.randn((10, 10)).to(torch.float32)], 1,), dict()),
      ("aten_cat_1", torch.ops.aten.cat, ([torch.randn((10, 10)).to(torch.float32)], 1,), dict()),
      ("aten_cat_2", torch.ops.aten.cat, ([torch.randn((10, 10)).to(torch.float32)], 1,), dict()),
      ("aten__cdist_forward_0", torch.ops.aten._cdist_forward, (rnd(torch.float32, (5, 7, 10)), rnd(torch.float32, (5, 8, 10)), 1.0, None,), dict()),
      ("aten_ceil_0", torch.ops.aten.ceil, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_clamp_0", torch.ops.aten.clamp, (rnd(torch.float32, (10, 10)), 0, 1,), dict()),
      ("aten_clamp_Tensor_0", torch.ops.aten.clamp.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (1,)),), dict()),
      ("aten_clone_0", torch.ops.aten.clone, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_constant_pad_nd_0", torch.ops.aten.constant_pad_nd, (rnd(torch.float32, (10, 10)), [0, 1], 1,), dict()),
      ("aten_convolution_0", torch.ops.aten.convolution, (rnd(torch.float32, (3, 2, 10)), rnd(torch.float32, (2, 2, 2)), None, [2], [0], [1], False, [0], 1,), dict()),
      ("aten_convolution_1", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1,), dict()),
      ("aten_convolution_2", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1,), dict()),
      ("aten_convolution_3", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1,), dict()),
      ("aten_convolution_4", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [2, 2], [2, 2], [1, 1], False, [0, 0], 1,), dict()),
      ("aten_convolution_5", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [1, 1], [0, 0], [2, 2], False, [0, 0], 1,), dict()),
      ("aten_convolution_6", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [2, 2], [0, 0], [2, 2], False, [0, 0], 1,), dict()),
      ("aten_convolution_7", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [2, 2], [1, 1], [2, 2], False, [0, 0], 1,), dict()),
      ("aten_convolution_8", torch.ops.aten.convolution, (rnd(torch.float32, (2, 6, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 2,), dict()),
      ("aten_convolution_9", torch.ops.aten.convolution, (rnd(torch.float32, (2, 6, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [2, 2], [0, 0], [1, 1], False, [0, 0], 2,), dict()),
      ("aten_convolution_10", torch.ops.aten.convolution, (rnd(torch.float32, (2, 6, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2,), dict()),
      ("aten_convolution_11", torch.ops.aten.convolution, (rnd(torch.float32, (2, 6, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [2, 2], [2, 2], [1, 1], False, [0, 0], 2,), dict()),
      ("aten_convolution_12", torch.ops.aten.convolution, (rnd(torch.float32, (2, 6, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [1, 1], [0, 0], [2, 2], False, [0, 0], 2,), dict()),
      ("aten_convolution_13", torch.ops.aten.convolution, (rnd(torch.float32, (2, 6, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [2, 2], [0, 0], [2, 2], False, [0, 0], 2,), dict()),
      ("aten_convolution_14", torch.ops.aten.convolution, (rnd(torch.float32, (2, 6, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), None, [2, 2], [1, 1], [2, 2], False, [0, 0], 2,), dict()),
      ("aten_convolution_15", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (3, 2, 2, 2)), None, [1, 1], [0, 0], [1, 1], True, [0, 0], 1,), dict()),
      ("aten_convolution_16", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (3, 2, 2, 2)), None, [2, 2], [0, 0], [1, 1], True, [0, 0], 1,), dict()),
      ("aten_convolution_17", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (3, 2, 2, 2)), None, [1, 1], [1, 1], [1, 1], True, [0, 0], 1,), dict()),
      ("aten_convolution_18", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (3, 2, 2, 2)), None, [2, 2], [2, 2], [1, 1], True, [0, 0], 1,), dict()),
      ("aten_convolution_19", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (3, 2, 2, 2)), None, [2, 2], [1, 1], [2, 2], True, [0, 0], 1,), dict()),
      ("aten_convolution_20", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), rnd(torch.float32, (2)), [1, 1], [0, 0], [1, 1], False, [0, 0], 1,), dict()),
      ("aten_convolution_21", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (3, 2, 2, 2)), rnd(torch.float32, (2)), [1, 1], [0, 0], [1, 1], True, [0, 0], 1,), dict()),
      ("aten_convolution_22", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), rnd(torch.float32, (2)), [1, 1], [1, 1], [1, 1], False, [0, 0], 1,), dict()),
      ("aten_convolution_23", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (3, 2, 2, 2)), rnd(torch.float32, (2)), [1, 1], [1, 1], [1, 1], True, [0, 0], 1,), dict()),
      ("aten_convolution_24", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), rnd(torch.float32, (2)), [1, 1], [1, 1], [2, 2], False, [0, 0], 1,), dict()),
      ("aten_convolution_25", torch.ops.aten.convolution, (rnd(torch.float32, (2, 3, 4, 4)), rnd(torch.float32, (3, 2, 2, 2)), rnd(torch.float32, (2)), [1, 1], [1, 1], [2, 2], True, [0, 0], 1,), dict()),
      ("aten_convolution_26", torch.ops.aten.convolution, (rnd(torch.float32, (2, 6, 4, 4)), rnd(torch.float32, (2, 3, 2, 2)), rnd(torch.float32, (2)), [1, 1], [1, 1], [1, 1], False, [0, 0], 2,), dict()),
      # # TODO(b/365559296): Add tests for output_padding.
      ("aten_copy_0", torch.ops.aten.copy, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10))), dict()),
      ("aten_copy_broadcast", torch.ops.aten.copy, (rnd(torch.float32, (10, 10)), torch.tensor(1.0, dtype=torch.float32)), dict()),
      ("aten_copy_cast_dtype", torch.ops.aten.copy, (rnd(torch.float32, (10, 10)), rnd(torch.int64, (10, 10))), dict()),
      ("aten_cos_0", torch.ops.aten.cos, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_cosh_0", torch.ops.aten.cosh, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_cumsum_0", torch.ops.aten.cumsum, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten_diagonal_0", torch.ops.aten.diagonal, (rnd(torch.float32, (10, 20)),), dict()),
      ("aten_div_Scalar_0", torch.ops.aten.div.Scalar, (rnd(torch.float32, (10, 10)), 0.5,), dict()),
      ("aten_div_Scalar_mode_0", torch.ops.aten.div.Scalar_mode, (rnd(torch.float32, (10, 10)), 0.123,), {"rounding_mode": "trunc"}),
      ("aten_div_Tensor_0", torch.ops.aten.div.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_div_Tensor_mode_0", torch.ops.aten.div.Tensor_mode, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), {"rounding_mode": "trunc"}),
      ("aten_embedding_0", torch.ops.aten.embedding, (rnd(torch.float32, (10, 10)), rnd(torch.int64, (10,)),), dict()),
      ("aten_eq_Scalar_2", torch.ops.aten.eq.Scalar, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten_eq_Tensor_0", torch.ops.aten.eq.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_erf_0", torch.ops.aten.erf, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_exp_0", torch.ops.aten.exp, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_expand_0", torch.ops.aten.expand, (rnd(torch.float32, (10, 1)), [10, 10],), dict()),
      ("aten_expand_copy_0", torch.ops.aten.expand_copy, (rnd(torch.float32, (10, 10)), [10, 10],), dict()),
      ("aten_expm1_0", torch.ops.aten.expm1, (rnd(torch.float32, (10, 10)),), dict()),
      # ("aten_fill_Scalar_0", torch.ops.aten.fill.Scalar, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_flip_0", torch.ops.aten.flip, (rnd(torch.float32, (10, 10)), [0, 1],), dict()),
      ("aten_floor_0", torch.ops.aten.floor, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_floor_1", torch.ops.aten.floor, (rnd(torch.float32, (10, 10), -10, 0),), dict()),
      ("aten_floor_2", torch.ops.aten.floor, (rnd(torch.float32, (10, 10), 0, 10),), dict()),
      ("aten_floor_3", torch.ops.aten.floor, (rnd(torch.float32, (10, 10), -100, 100),), dict()),
      ("aten_fmod_Scalar_0", torch.ops.aten.fmod.Scalar, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_fmod_Tensor_0", torch.ops.aten.fmod.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_full_0", torch.ops.aten.full, ((5, 5), 0.123,), dict()),
      ("aten_full_float32", torch.ops.aten.full, ((5, 5), 0.123,), dict(dtype=torch.float32)),
      ("aten_full_int64", torch.ops.aten.full, ((5, 5), 10), dict(dtype=torch.int64)),
      ("aten_full_like_0", torch.ops.aten.full_like, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_gather_0", torch.ops.aten.gather, (rnd(torch.float32, (10, 10)), 1, rnd(torch.int64, (2, 2)),), dict()),
      ("aten_ge_Scalar_0", torch.ops.aten.ge.Scalar, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_ge_Tensor_0", torch.ops.aten.ge.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_gelu_0", torch.ops.aten.gelu, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_glu_0", torch.ops.aten.glu, (rnd(torch.float32, (10, 10)), 0,), dict()),
      # ("aten_grid_sampler_2d_0", torch.ops.aten.grid_sampler_2d, (rnd(torch.float32, (1, 3, 2, 10)), rnd(torch.float32, (1, 2, 2, 2)), 0, 0, False,), dict()),
      ("aten_gt_Scalar_0", torch.ops.aten.gt.Scalar, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_gt_Tensor_0", torch.ops.aten.gt.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_hardtanh_0", torch.ops.aten.hardtanh, (rnd(torch.float32, (10, 10)), 1,), dict()),
      # ("aten_index_put_0", torch.ops.aten.index_put, (rnd(torch.float32, (10, 10)), [torch.randint(0, 10, (1,)).to(torch.int64)], rnd(torch.float32, (10,)),), dict()),
      ("aten_index_select_int64_index", torch.ops.aten.index_select, (rnd(torch.float32, (2, 10)), 1, rnd(torch.int64, (2,)),), dict()),
      ("aten_index_select_int32_index", torch.ops.aten.index_select, (rnd(torch.float32, (2, 10)), 1, rnd(torch.int32, (2,)),), dict()),
      ("aten_index_Tensor_0", torch.ops.aten.index.Tensor, (rnd(torch.float32, (10, 10)), [torch.randint(0, 10, (2,)).to(torch.int64)],), dict()),
      ("aten_isinf_0", torch.ops.aten.isinf, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_isnan_0", torch.ops.aten.isnan, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_le_Scalar_0", torch.ops.aten.le.Scalar, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_le_Tensor_0", torch.ops.aten.le.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_leaky_relu_0", torch.ops.aten.leaky_relu, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten_lift_fresh_copy_0", torch.ops.aten.lift_fresh_copy, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_log_0", torch.ops.aten.log, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_log10_0", torch.ops.aten.log10, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_log1p_0", torch.ops.aten.log1p, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_log2_0", torch.ops.aten.log2, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten__log_softmax_0", torch.ops.aten._log_softmax, (rnd(torch.float32, (10, 10)), 1, False,), dict()),
      ("aten_logical_and_0", torch.ops.aten.logical_and, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_logical_not_0", torch.ops.aten.logical_not, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_logical_or_0", torch.ops.aten.logical_or, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_logical_xor_0", torch.ops.aten.logical_xor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_logit_0", torch.ops.aten.logit, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_lt_Scalar_0", torch.ops.aten.lt.Scalar, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_lt_Tensor_0", torch.ops.aten.lt.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      # ("aten_masked_fill_Scalar_0", torch.ops.aten.masked_fill.Scalar, (rnd(torch.float32, (10, 10)), rnd(torch.bool, (10, 10)), 0.123,), dict()),
      ("aten_max_dim_0", torch.ops.aten.max.dim, (rnd(torch.float32, (10, 10)), 1,), dict()),
      # ("aten_max_pool2d_with_indices_0", torch.ops.aten.max_pool2d_with_indices, (rnd(torch.float32, (3, 2, 10)), [2, 2], [1, 1], [1, 1],), dict()),
      # ("aten_max_pool2d_with_indices_2", torch.ops.aten.max_pool2d_with_indices, (torch.arange(0, 60).reshape(3, 2, 10), [2, 2], [1, 1], [1, 1],), dict()),
      # ("aten_max_pool3d_with_indices_0", torch.ops.aten.max_pool3d_with_indices, (rnd(torch.float32, (1, 3, 2, 10)), [2, 2, 2], [1, 1, 1], [1, 1, 1],), dict()),
      # ("aten_max_pool3d_with_indices_2", torch.ops.aten.max_pool3d_with_indices, (torch.arange(0, 60).reshape(1, 3, 2, 10), [2, 2, 2], [1, 1, 1], [1, 1, 1],), dict()),
      ("aten_maximum_0", torch.ops.aten.maximum, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_mean_0", torch.ops.aten.mean, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_mean_dim_0", torch.ops.aten.mean.dim, (rnd(torch.float32, (10, 10)), None,), dict()),
      # ("aten_min_dim_0", torch.ops.aten.min.dim, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten_minimum_0", torch.ops.aten.minimum, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_mm_0", torch.ops.aten.mm, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_mul_Scalar_0", torch.ops.aten.mul.Scalar, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_mul_Tensor_0", torch.ops.aten.mul.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      # ("aten__native_batch_norm_legit_0", torch.ops.aten._native_batch_norm_legit, (rnd(torch.float32, (10, 10)), None, None, rnd(torch.float32, (10,)), rnd(torch.float32, (10,)), False, 1.0, 1.0,), dict()),
      ("aten__native_batch_norm_legit_no_stats_0", torch.ops.aten._native_batch_norm_legit.no_stats, (rnd(torch.float32, (1, 3, 2, 10)), rnd(torch.float32, (1, 3, 1, 1)), rnd(torch.float32, (1, 3, 1, 1)), True, 0.0, 1.0,), dict()),
      ("aten__native_batch_norm_legit_no_training_0", torch.ops.aten._native_batch_norm_legit_no_training, (rnd(torch.float32, (10, 10)), None, None, rnd(torch.float32, (10,)), rnd(torch.float32, (10,)), 1.0, 1.0,), dict()),
      # ("aten_native_dropout_0", torch.ops.aten.native_dropout, (rnd(torch.float32, (10, 10)), 1.0, True,), dict()),
      ("aten_native_group_norm_0", torch.ops.aten.native_group_norm, (rnd(torch.float32, (1, 3, 2, 10)), None, None, 1, 3, 20, 1, 0.0,), dict()),
      ("aten_native_layer_norm_0", torch.ops.aten.native_layer_norm, (rnd(torch.float32, (1, 3, 2, 10)), [1, 3, 2, 10], None, None, 0.0,), dict()),
      ("aten_native_layer_norm_1", torch.ops.aten.native_layer_norm, (rnd(torch.float32, (1, 3, 2, 10)), [3, 2, 10], None, None, 0.0,), dict()),
      ("aten_native_layer_norm_2", torch.ops.aten.native_layer_norm, (rnd(torch.float32, (2, 3, 2, 10)), [2, 10], None, None, 0.0,), dict()),
      ("aten_native_layer_norm_3", torch.ops.aten.native_layer_norm, (rnd(torch.float32, (2, 3, 2, 10)), [3, 2, 10], rnd(torch.float32, (3, 2, 10)), rnd(torch.float32, (3, 2, 10)), 0.0,), dict()),
      ("aten_native_layer_norm_4", torch.ops.aten.native_layer_norm, (rnd(torch.float32, (2, 3, 2, 10)), [3, 2, 10], rnd(torch.float32, (3, 2, 10)), rnd(torch.float32, (3, 2, 10)), 2,), dict()),
      ("aten_native_layer_norm_5", torch.ops.aten.native_layer_norm, (rnd(torch.float32, (2, 3, 2, 4, 5)), [3, 2, 4, 5], rnd(torch.float32, (3, 2, 4, 5)), rnd(torch.float32, (3, 2, 4, 5)), 0.1,), dict()),
      ("aten_native_layer_norm_6", torch.ops.aten.native_layer_norm, (rnd(torch.float32, (2, 3, 2, 4, 5)), [2, 4, 5], rnd(torch.float32, (2, 4, 5)), rnd(torch.float32, (2, 4, 5)), 0.1,), dict()),
      ("aten_ne_Scalar_2", torch.ops.aten.ne.Scalar, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten_ne_Tensor_0", torch.ops.aten.ne.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_neg_0", torch.ops.aten.neg, (rnd(torch.float32, (10, 10)),), dict()),
      # ("aten_nonzero_2", torch.ops.aten.nonzero, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten__pdist_forward_0", torch.ops.aten._pdist_forward, (rnd(torch.float32, (10, 10)), 1.0,), dict()),
      ("aten_permute_0", torch.ops.aten.permute, (rnd(torch.float32, (10, 10)), [0, 1],), dict()),
      ("aten_permute_copy_0", torch.ops.aten.permute_copy, (rnd(torch.float32, (2, 2, 2)), [1, 2, 0],), dict()),
      ("aten_pixel_shuffle_0", torch.ops.aten.pixel_shuffle, (rnd(torch.float32, (1, 3, 10, 10)), 1,), dict()),
      ("aten_pow_Scalar_0", torch.ops.aten.pow.Scalar, (1.123, rnd(torch.float32, (10, 10)),), dict()),
      ("aten_pow_Tensor_Scalar_0", torch.ops.aten.pow.Tensor_Scalar, (rnd(torch.float32, (10, 10)), 1.2,), dict()),
      ("aten_pow_Scalar_1", torch.ops.aten.pow.Scalar, (10000, torch.randn(16 * 8),), dict()),
      ("aten_pow_Tensor_Tensor_0", torch.ops.aten.pow.Tensor_Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_prod_0", torch.ops.aten.prod, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_prod_dim_int_0", torch.ops.aten.prod.dim_int, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten_reciprocal_0", torch.ops.aten.reciprocal, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_reflection_pad1d_0", torch.ops.aten.reflection_pad1d, (rnd(torch.float32, (10, 10)), [0, 1],), dict()),
      ("aten_reflection_pad2d_0", torch.ops.aten.reflection_pad2d, (rnd(torch.float32, (3, 2, 10)), [1, 1, 1, 1],), dict()),
      ("aten_reflection_pad3d_0", torch.ops.aten.reflection_pad3d, (rnd(torch.float32, (3, 3, 3, 3)), [1, 2, 1, 2, 1, 2],), dict()),
      ("aten_relu_0", torch.ops.aten.relu, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_remainder_Scalar_2", torch.ops.aten.remainder.Scalar, (rnd(torch.float32, (10, 10)), 2,), dict()),
      ("aten_remainder_Tensor_0", torch.ops.aten.remainder.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      # ("aten_replication_pad2d_0", torch.ops.aten.replication_pad2d, (rnd(torch.float32, (3, 2, 10)), [1, 1, 1, 1],), dict()),
      # ("aten_replication_pad3d_0", torch.ops.aten.replication_pad3d, (rnd(torch.float32, (1, 3, 2, 10)), [1, 1, 1, 1, 1, 1],), dict()),
      ("aten_roll_0", torch.ops.aten.roll, (rnd(torch.float32, (10, 10)), [0, 1], [0, 1],), dict()),
      ("aten_round_0", torch.ops.aten.round, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_rsqrt_0", torch.ops.aten.rsqrt, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_rsub_Scalar_0", torch.ops.aten.rsub.Scalar, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_scatter_add_0", torch.ops.aten.scatter_add, (rnd(torch.float32, (10, 10)), 1, rnd(torch.int64, (2, 2)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_scatter_reduce_two_0", torch.ops.aten.scatter_reduce.two, (rnd(torch.float32, (10, 10)), 1, rnd(torch.int64, (10, 10)), rnd(torch.float32, (10, 10)), "sum",), dict()),
      ("aten_scatter_src_0", torch.ops.aten.scatter.src, (rnd(torch.float32, (10, 10)), 1, rnd(torch.int64, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_scatter_value_0", torch.ops.aten.scatter.value, (rnd(torch.float32, (10, 10)), 1, rnd(torch.int64, (10, 10)), 1,), dict()),
      ("aten_select_copy_int_0", torch.ops.aten.select_copy.int, (rnd(torch.float32, (10, 10)), 1, 0,), dict()),
      ("aten_select_int_0", torch.ops.aten.select.int, (rnd(torch.float32, (10, 10)), 1, 1,), dict()),
      ("aten_select_scatter_0", torch.ops.aten.select_scatter, (rnd(torch.float32, (10, 10)), rnd(torch.int64, (10,)), 1, 0,), dict()),
      ("aten_sigmoid_0", torch.ops.aten.sigmoid, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_sign_0", torch.ops.aten.sign, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_sin_0", torch.ops.aten.sin, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_sinh_0", torch.ops.aten.sinh, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_slice_copy_Tensor_0", torch.ops.aten.slice_copy.Tensor, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_slice_scatter_0", torch.ops.aten.slice_scatter, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten_slice_scatter_1", torch.ops.aten.slice_scatter, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (4, 10)), 0, 1, 5), dict()),
      ("aten_slice_scatter_2", torch.ops.aten.slice_scatter, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (3, 10)), 0, -6, -3), dict()),
      ("aten_slice_scatter_3", torch.ops.aten.slice_scatter, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (3, 10)), 0, 2, 8, 2), dict()),
      ("aten_slice_scatter_4", torch.ops.aten.slice_scatter, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (2, 10)), 0, 2, 7, 3), dict()),
      ("aten_slice_scatter_5", torch.ops.aten.slice_scatter, (rnd(torch.float32, (0, 10)), rnd(torch.float32, (0, 3)), 1, 0, -7), dict()),
      ("aten_slice_scatter_6", torch.ops.aten.slice_scatter, (rnd(torch.float32, (8, 3, 3)), rnd(torch.float32, (0, 3, 3)), 0, -8, 0), dict()),
      ("aten_slice_Tensor_0", torch.ops.aten.slice.Tensor, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten__softmax_0", torch.ops.aten._softmax, (rnd(torch.float32, (10, 10)), 1, False,), dict()),
      ("aten_split_copy_Tensor_0", torch.ops.aten.split_copy.Tensor, (rnd(torch.float32, (10, 10)), 2,), dict()),
      ("aten_split_with_sizes_0", torch.ops.aten.split_with_sizes, (rnd(torch.float32, (10, 10)), [1, 2, 3, 4],), dict()),
      ("aten_sqrt_0", torch.ops.aten.sqrt, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_squeeze_copy_dim_0", torch.ops.aten.squeeze_copy.dim, (rnd(torch.float32, (10, 10)), 0,), dict()),
      ("aten_squeeze_dims_0", torch.ops.aten.squeeze.dims, (rnd(torch.float32, (10, 10)), [0, 1],), dict()),
      # ("aten_stack_0", torch.ops.aten.stack, ([torch.randn((10, 10)).to(torch.float32), torch.randn((10, 10)).to(torch.float32), torch.randn((10, 10)).to(torch.float32)],), dict()),
      # ("aten_stack_1", torch.ops.aten.stack, ([torch.randn((10, 10)).to(torch.float32), torch.randn((10, 10)).to(torch.float32), torch.randn((10, 10)).to(torch.float32)],), dict()),
      # ("aten_stack_2", torch.ops.aten.stack, ([torch.randn((10, 10)).to(torch.float32), torch.randn((10, 10)).to(torch.float32), torch.randn((10, 10)).to(torch.float32)],), dict()),
      ("aten_sub_Scalar_0", torch.ops.aten.sub.Scalar, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_sub_Tensor_0", torch.ops.aten.sub.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_sum_dim_IntList_0", torch.ops.aten.sum.dim_IntList, (rnd(torch.float32, (10, 10)), None,), dict()),
      ("aten_tan_0", torch.ops.aten.tan, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_tanh_0", torch.ops.aten.tanh, (rnd(torch.float32, (10, 10)),), dict()),
      # ("aten_topk_0", torch.ops.aten.topk, (torch.arange(0, 100).reshape(10, 10).to(torch.float32), 1, 1, False, False,), dict()),
      # ("aten_topk_1", torch.ops.aten.topk, (rnd(torch.float32, (10, 10)), 1, 1, True, False,), dict()),
      ("aten_transpose_copy_int_0", torch.ops.aten.transpose_copy.int, (rnd(torch.float32, (10, 10)), 0, 1,), dict()),
      ("aten_tril_0", torch.ops.aten.tril, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_trunc_0", torch.ops.aten.trunc, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_unbind_copy_int_0", torch.ops.aten.unbind_copy.int, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_unsqueeze_copy_0", torch.ops.aten.unsqueeze_copy, (rnd(torch.float32, (2, 0, 2)), 1,), dict()),
      ("aten_upsample_bilinear2d_0", torch.ops.aten.upsample_bilinear2d, (rnd(torch.float32, (1, 3, 2, 10)), [3, 20], False,), dict()),
      ("aten_upsample_nearest2d_0", torch.ops.aten.upsample_nearest2d, (rnd(torch.float32, (1, 3, 2, 10)), [3, 20],), dict()),
      # ("aten_var_correction_0", torch.ops.aten.var.correction, (rnd(torch.float32, (10, 10)),), dict()),
      # ("aten_var_correction_2", torch.ops.aten.var.correction, (rnd(torch.float32, (10, 10)),), dict(correction=0)),
      ("aten_view_0", torch.ops.aten.view, (rnd(torch.float32, (10, 10)), [1, 100],), dict()),
      ("aten_view_copy_0", torch.ops.aten.view_copy, (rnd(torch.float32, (10, 10)), [2, 5, 10],), dict()),
      ("aten_where_self_0", torch.ops.aten.where.self, (rnd(torch.bool, (10, 10)), rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      # fmt: on
      # pyformat: enable
  )
  def test_op(self, op, args, kwargs):
    self._run_export_and_compare(op, args, kwargs)

  @googletest.skip("wip jax lowering")
  def test_aten_native_batch_norm_legit(self):
    batch = 3
    channel = 2
    args = (
        torch.randn((batch, channel, 2, 2)).to(torch.float32),
        torch.ones(channel),
        torch.zeros(channel),
        torch.zeros(channel),
        torch.ones(channel),
        False,
        0.5,
        1,
    )
    kwargs = dict()
    self._run_export_and_compare(
        torch.ops.aten._native_batch_norm_legit, args, kwargs
    )

  @googletest.skip("wip jax lowering")
  def test_aten_native_batch_norm_legit_none(self):
    batch = 3
    channel = 2
    args = (
        torch.randn((batch, channel, 4, 4)).to(torch.float32),
        None,
        None,
        torch.ones(channel),
        torch.zeros(channel),
        False,
        0.5,
        1,
    )
    kwargs = dict()
    self._run_export_and_compare(
        torch.ops.aten._native_batch_norm_legit, args, kwargs
    )

  @googletest.skip("wip jax lowering")
  def test_aten_native_batch_norm_legit_training_none(self):
    batch = 3
    channel = 2
    args = (
        torch.randn((batch, channel, 4, 3)).to(torch.float32),
        None,
        None,
        torch.zeros(channel),
        torch.ones(channel),
        True,
        0.2,
        2e-05,
    )
    kwargs = dict()
    self._run_export_and_compare(
        torch.ops.aten._native_batch_norm_legit, args, kwargs
    )

  def test_aten_native_batch_norm_legit_no_training(self):
    batch = 3
    channel = 2
    args = (
        torch.randn((batch, channel, 4, 3)).to(torch.float32),
        torch.ones(channel),
        torch.zeros(channel),
        torch.zeros(channel),
        torch.ones(channel),
        0.2,
        2e-05,
    )
    kwargs = dict()
    self._run_export_and_compare(
        torch.ops.aten._native_batch_norm_legit_no_training, args, kwargs
    )

  @googletest.skip("wip jax lowering")
  def test_aten_native_batch_norm_training(self):
    batch = 3
    channel = 2
    args = (
        torch.randn((batch, channel, 4, 3)).to(torch.float32),
        torch.ones(channel),
        torch.zeros(channel),
        torch.zeros(channel),
        torch.ones(channel),
        True,
        0.1,
        1e-05,
    )
    kwargs = dict()
    self._run_export_and_compare(torch.ops.aten.native_batch_norm, args, kwargs)

  @googletest.skip("wip jax lowering")
  def test_aten_native_batch_norm_training_none(self):
    batch = 3
    channel = 2
    args = (
        torch.randn((batch, channel, 4, 3)).to(torch.float32),
        None,
        None,
        torch.zeros(channel),
        torch.ones(channel),
        True,
        0.1,
        1e-05,
    )
    kwargs = dict()
    self._run_export_and_compare(torch.ops.aten.native_batch_norm, args, kwargs)

  def test_aten_native_batch_norm_eval(self):
    batch = 3
    channel = 2
    args = (
        torch.randn((batch, channel, 4, 3)).to(torch.float32),
        torch.ones(channel),
        torch.zeros(channel),
        torch.zeros(channel),
        torch.ones(channel),
        False,
        0.2,
        2e-05,
    )
    kwargs = dict()
    self._run_export_and_compare(torch.ops.aten.native_batch_norm, args, kwargs)


if __name__ == "__main__":
  googletest.main()
