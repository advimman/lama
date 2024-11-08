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
from typing import Callable, Union

from ai_edge_torch import config
from ai_edge_torch import fx_pass_base
from ai_edge_torch import lowertools
from ai_edge_torch.generative.fx_passes import CanonicalizePass
from ai_edge_torch.generative.fx_passes import RemoveSDPACompositeZeroMaskPass
from ai_edge_torch.generative.layers.attention import SelfAttention
import ai_edge_torch.generative.layers.model_config as layers_cfg
import ai_edge_torch.generative.layers.unet.model_config as unet_cfg
import torch

from absl.testing import absltest as googletest


def _export_to_stablehlo(func: Union[torch.nn.Module, Callable], export_args):
  if not isinstance(func, torch.nn.Module):

    class TestModule(torch.nn.Module):

      def forward(self, *args, **kwargs):
        return func(*args, **kwargs)

    module = TestModule().eval()
  else:
    module = func

  exported_program = torch.export.export(module, export_args)
  exported_program = fx_pass_base.run_passes(
      exported_program,
      [
          RemoveSDPACompositeZeroMaskPass(),
          CanonicalizePass(),
      ],
  )

  return lowertools.exported_program_to_mlir_text(exported_program)


class TestRemoveSDPAZeroMaskPass(googletest.TestCase):

  def test_self_attention_no_zero_mask_composite_input(self):
    class SampleSdpaBlock(torch.nn.Module):
      """Sample attention block with SDPA"""

      def __init__(self, config: unet_cfg.AttentionBlock2DConfig):
        super().__init__()
        self.config = config
        self.attention = SelfAttention(
            config.attention_batch_size,
            config.dim,
            config.attention_config,
            enable_hlfb=config.enable_hlfb,
        )

      def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = input_tensor.shape
        x = input_tensor
        x = input_tensor.view(B, C, H * W)
        x = x.transpose(-1, -2)
        # x = x.contiguous()  # Prevent BATCH_MATMUL op in converted tflite.
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view(B, C, H, W)
        return x

    def get_model_config() -> unet_cfg.AttentionBlock2DConfig:
      """Get configs for the Decoder of Stable Diffusion v1.5"""
      in_channels = 3
      latent_channels = 4
      out_channels = 3
      block_out_channels = [128, 256, 512, 512]
      scaling_factor = 0.18215
      layers_per_block = 3

      norm_config = layers_cfg.NormalizationConfig(
          layers_cfg.NormalizationType.GROUP_NORM, group_num=32
      )

      return unet_cfg.AttentionBlock2DConfig(
          dim=block_out_channels[-1],
          normalization_config=norm_config,
          attention_config=layers_cfg.AttentionConfig(
              num_heads=1,
              head_dim=block_out_channels[-1],
              num_query_groups=1,
              qkv_use_bias=True,
              output_proj_use_bias=True,
              enable_kv_cache=False,
              qkv_transpose_before_split=True,
              rotary_percentage=0.0,
          ),
      )

    stablehlo = _export_to_stablehlo(
        SampleSdpaBlock(get_model_config()).eval(),
        (torch.rand(1, 512, 64, 64),),
    )

    if config.Config.use_torch_xla:
      self.assertTrue(
          re.search(
              'stablehlo\.composite "odml\.scaled_dot_product_attention" %\d+,'
              ' %\d+, %\d+ {',
              stablehlo,
          )
      )
    else:
      self.assertEqual(stablehlo.count('stablehlo.custom_call @mark_tensor'), 4)


if __name__ == '__main__':
  googletest.main()
