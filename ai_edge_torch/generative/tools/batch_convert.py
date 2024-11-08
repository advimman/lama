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

"""A python script to convert a batch of Generative models to TF Lite."""

import dataclasses
import enum
import logging
import os
import pathlib
from typing import Callable, Sequence

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.gemma import gemma1
from ai_edge_torch.generative.examples.gemma import gemma2
from ai_edge_torch.generative.examples.llama import llama
from ai_edge_torch.generative.examples.openelm import openelm
from ai_edge_torch.generative.examples.phi import phi2
from ai_edge_torch.generative.examples.phi import phi3
from ai_edge_torch.generative.examples.qwen import qwen
from ai_edge_torch.generative.examples.smollm import smollm
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.utilities import converter
import torch

_CHECKPOINT_ROOT_PATH = flags.DEFINE_string(
    "checkpoint_root_path",
    os.path.join(pathlib.Path.home(), "Downloads/llm_data/"),
    "The root path to the checkpoints.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    os.path.join(pathlib.Path.home(), "models"),
    "The output directory to store the converted models.",
)

_MODELS = flags.DEFINE_list(
    "models",
    [
        "gemma",
        "gemma2",
        "llama3.2",
        "openelm",
        "phi2",
        "phi3.5",
        "qwen2.5",
        "smollm",
        "tinyllama",
    ],
    "The list of models to convert.",
)

_PREFILL_SEQ_LEN = flags.DEFINE_integer(
    "prefill_seq_len",
    1024,
    "The maximum size of prefill input tensor.",
)

_KV_CACHE_MAX_LEN = flags.DEFINE_integer(
    "kv_cache_max_len",
    1280,
    "The maximum size of KV cache buffer, including both prefill and decode.",
)

_PRECISIONS = flags.DEFINE_list(
    "precisions",
    ["q8", "f32"],
    "The list of precisions to convert.",
)


@enum.unique
class ExportPrecision(enum.Enum):
  """Specifies the precision of the exported model."""

  INT8 = enum.auto()
  FP32 = enum.auto()


@dataclasses.dataclass
class ConversionConfig:
  """A dataclass to store the conversion config for a model."""

  model_name: str
  input_checkpoint: str
  tflite_output_path: str
  prefill_seq_len: int
  kv_cache_max_len: int
  export_precision: Sequence[ExportPrecision]
  model_builder: Callable[..., torch.nn.Module]
  model_size: str

  def print_config(self) -> None:
    """Prints the conversion config."""
    logging.info("Model name: %s", self.model_name)
    logging.info("Input checkpoint: %s", self.input_checkpoint)
    logging.info("TF Lite output path: %s", self.tflite_output_path)
    logging.info("Prefill seq len: %s", self.prefill_seq_len)
    logging.info("KV cache max len: %s", self.kv_cache_max_len)
    logging.info("Export precision: %s", self.export_precision)
    logging.info("Model size: %s", self.model_size)


def get_conversion_config(
    model_name: str,
    input_checkpoint_subdir: str,
    tflite_output_subdir: str,
    model_builder: Callable[..., torch.nn.Module],
    model_size: str,
) -> ConversionConfig:
  """Returns the conversion config for a model."""
  export_precision = []
  if "q8" in _PRECISIONS.value:
    export_precision.append(ExportPrecision.INT8)
  if "f32" in _PRECISIONS.value:
    export_precision.append(ExportPrecision.FP32)

  return ConversionConfig(
      model_name=model_name,
      input_checkpoint=os.path.join(
          _CHECKPOINT_ROOT_PATH.value, input_checkpoint_subdir
      ),
      tflite_output_path=os.path.join(_OUTPUT_DIR.value, tflite_output_subdir),
      prefill_seq_len=_PREFILL_SEQ_LEN.value,
      kv_cache_max_len=_KV_CACHE_MAX_LEN.value,
      export_precision=export_precision,
      model_builder=model_builder,
      model_size=model_size,
  )


def prepare_conversion_configs() -> Sequence[ConversionConfig]:
  """Prepares the conversion configs according to the flags."""
  conversion_configs = []
  for model in _MODELS.value:
    if model == "gemma":
      conversion_configs.append(
          get_conversion_config(
              model_name="gemma",
              input_checkpoint_subdir="gemma-2b",
              tflite_output_subdir="gemma",
              model_builder=gemma1.build_2b_model,
              model_size="2b",
          )
      )
    elif model == "gemma2":
      conversion_configs.append(
          get_conversion_config(
              model_name="gemma2",
              input_checkpoint_subdir="gemma2-2b",
              tflite_output_subdir="gemma2",
              model_builder=gemma2.build_2b_model,
              model_size="2b",
          )
      )
    elif model == "llama3.2":
      conversion_configs.append(
          get_conversion_config(
              model_name="llama3.2",
              input_checkpoint_subdir="llama",
              tflite_output_subdir="llama",
              model_builder=llama.build_3b_model,
              model_size="3b",
          )
      )
    elif model == "openelm":
      conversion_configs.append(
          get_conversion_config(
              model_name="openelm",
              input_checkpoint_subdir="openelm",
              tflite_output_subdir="openelm",
              model_builder=openelm.build_model,
              model_size="3b",
          )
      )
    elif model == "phi2":
      conversion_configs.append(
          get_conversion_config(
              model_name="phi2",
              input_checkpoint_subdir="phi2",
              tflite_output_subdir="phi2",
              model_builder=phi2.build_model,
              model_size="2.7b",
          )
      )
    elif model == "phi3.5":
      conversion_configs.append(
          get_conversion_config(
              model_name="phi3.5",
              input_checkpoint_subdir="phi3",
              tflite_output_subdir="phi3",
              model_builder=phi3.build_model,
              model_size="3.8b",
          )
      )
    elif model == "qwen2.5":
      conversion_configs.append(
          get_conversion_config(
              model_name="qwen2.5",
              input_checkpoint_subdir="qwen",
              tflite_output_subdir="qwen",
              model_builder=qwen.build_3b_model,
              model_size="3b",
          )
      )
    elif model == "smollm":
      conversion_configs.append(
          get_conversion_config(
              model_name="smollm",
              input_checkpoint_subdir="smollm",
              tflite_output_subdir="smollm",
              model_builder=smollm.build_model,
              model_size="135m",
          )
      )
    elif model == "tinyllama":
      conversion_configs.append(
          get_conversion_config(
              model_name="tinyllama",
              input_checkpoint_subdir="tiny_llama",
              tflite_output_subdir="tiny_llama",
              model_builder=tiny_llama.build_model,
              model_size="1.1b",
          )
      )
    else:
      raise ValueError(f"Unsupported model: {model}")
  return conversion_configs


def get_output_filename(
    model_name: str,
    model_size: str,
    precision: ExportPrecision,
    prefill_seq_len: int,
    kv_cache_max_len: int,
) -> str:
  """Returns the output filename for a converted TF Litemodel."""
  if precision == ExportPrecision.INT8:
    precision_str = "q8"
  elif precision == ExportPrecision.FP32:
    precision_str = "f32"
  else:
    raise ValueError(f"Unsupported precision: {precision}")
  return f"{model_name}_{model_size}_{precision_str}_seq{prefill_seq_len}_ekv{kv_cache_max_len}.tflite"


def convert_models(conversion_configs: Sequence[ConversionConfig]) -> None:
  """Executes the conversion for a batch of models specified by the `conversion_configs`."""
  for config in conversion_configs:
    logging.info(
        "Converting model: %s with the following config:", config.model_name
    )
    config.print_config()
    pytorch_model = config.model_builder(
        config.input_checkpoint, kv_cache_max_len=config.kv_cache_max_len
    )
    for precision in config.export_precision:
      output_filename = get_output_filename(
          config.model_name,
          config.model_size,
          precision,
          config.prefill_seq_len,
          config.kv_cache_max_len,
      )
      converter.convert_to_tflite(
          pytorch_model,
          tflite_path=os.path.join(config.tflite_output_path, output_filename),
          prefill_seq_len=config.prefill_seq_len,
          quantize=True if precision == ExportPrecision.INT8 else False,
      )
      logging.info("Successfully converted model: %s", output_filename)


def main(_):
  convert_models(prepare_conversion_configs())


if __name__ == "__main__":
  app.run(main)
