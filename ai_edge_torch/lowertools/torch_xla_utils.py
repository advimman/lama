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

import copy
import dataclasses
from dataclasses import dataclass
import gc
import itertools
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

if "PJRT_DEVICE" not in os.environ:
  # https://github.com/google-ai-edge/ai-edge-torch/issues/326
  os.environ["PJRT_DEVICE"] = "CPU"

from ai_edge_torch import model
from ai_edge_torch._convert import conversion_utils
from ai_edge_torch._convert import signature as signature_module
from ai_edge_torch.lowertools import common_utils
from ai_edge_torch.lowertools import translate_recipe
from ai_edge_torch.quantize import quant_config as qcfg
import torch
from torch_xla import stablehlo

try:
  import tensorflow as tf

  from tensorflow.compiler.tf2xla.python import xla as tfxla

  from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb  # isort:skip
except ImportError:
  logging.error(
      "This module needs tensorflow with xla support.\n"
      "Please install tensorflow with `pip install tf-nightly`.\n"
  )
  raise

MlirBundle = stablehlo.StableHLOModelBundle


@dataclasses.dataclass
class MergedBundle:

  bundle: stablehlo.StableHLOModelBundle
  exported_programs: list[torch.export.ExportedProgram]
  deduped_tf_vars: list[tf.Variable]


def exported_program_to_mlir(
    exported_program: torch.export.ExportedProgram,
    sample_args: tuple[torch.Tensor],
) -> stablehlo.StableHLOModelBundle:
  # Setting export_weights to False here so that pytorch/xla avoids copying the
  # weights to a numpy array which would lead to memory bloat. This means that
  # the state_dict in the returned bundle is going to be empty.
  return stablehlo.exported_program_to_stablehlo(
      exported_program,
      stablehlo.StableHLOExportOptions(
          override_tracing_arguments=sample_args, export_weights=False
      ),
  )._bundle


def merge_mlir_bundles(
    bundles: list[stablehlo.StableHLOModelBundle],
    signatures: list[signature_module.Signature],
    exported_programs: list[torch.export.ExportedProgram],
) -> stablehlo.StableHLOGraphModule:
  state_dict, deduped_tf_vars = common_utils.gather_state_dict(
      exported_programs, signatures
  )

  new_shlo_model_bundle = stablehlo.StableHLOModelBundle(
      state_dict=state_dict, additional_constants=[], stablehlo_funcs=[]
  )

  for bundle, signature in zip(bundles, signatures):
    const_offset = len(new_shlo_model_bundle.additional_constants)
    for func in bundle.stablehlo_funcs:
      func.meta.name = signature.name + "_" + func.meta.name
      for loc in func.meta.input_locations:
        if loc.type_ == stablehlo.VariableType.CONSTANT:
          loc.position += const_offset
        elif loc.type_ == stablehlo.VariableType.PARAMETER:
          loc.name = signature.name + "_" + loc.name
      new_shlo_model_bundle.stablehlo_funcs.append(func)
    new_shlo_model_bundle.additional_constants.extend(
        bundle.additional_constants
    )
  return MergedBundle(
      bundle=new_shlo_model_bundle,
      exported_programs=exported_programs,
      deduped_tf_vars=deduped_tf_vars,
  )


def _get_shape_with_dynamic(signature: stablehlo.VariableSignature):
  shape = copy.copy(signature.shape)
  for i in signature.dynamic_dims:
    shape[i] = None
  return shape


def _wrap_as_tf_func(
    func: stablehlo.StableHLOFunc,
    bundle: stablehlo.StableHLOModelBundle,
    exported_program: torch.export.ExportedProgram,
):
  def inner(*args):
    type_info = [sig.dtype for sig in func.meta.output_signature]
    shape_info = [
        _get_shape_with_dynamic(sig) for sig in func.meta.output_signature
    ]
    call_args = stablehlo._extract_call_parameters(args, func.meta, bundle)
    call_module_return = tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=type_info,
        Sout=shape_info,
        function_list=[],
        module=func.bytecode,
    )
    spec = exported_program.call_spec.out_spec

    # The module returning a flat array.
    if not spec.context:
      return call_module_return

    flat_names = common_utils.flat_dict_names(spec.children_specs, spec.context)
    return {name: value for name, value in zip(flat_names, call_module_return)}

  return inner


def _make_tf_signature(
    meta: stablehlo.StableHLOFunctionMeta,
    signature: signature_module.Signature,
) -> list[tf.TensorSpec]:
  input_names = signature.flat_arg_names
  input_pos_to_spec = {
      loc.position: spec
      for loc, spec in itertools.chain(
          zip(meta.input_locations, meta.input_signature), meta.unused_inputs
      )
      if loc.type_ == stablehlo.VariableType.INPUT_ARG
  }
  assert len(input_pos_to_spec) == len(input_names)

  primitive_type_to_tf_type = {"int": "int32", "float": "float32"}
  ret: list[tf.TensorSpec] = []
  for i, name in enumerate(input_names):
    spec = input_pos_to_spec[i]
    shape = _get_shape_with_dynamic(spec)
    ret.append(
        tf.TensorSpec(
            shape=shape,
            dtype=primitive_type_to_tf_type[spec.dtype]
            if spec.dtype in primitive_type_to_tf_type
            else spec.dtype,
            name=name,
        )
    )
  return ret


def exported_program_to_mlir_text(
    exported_program: torch.export.ExportedProgram,
) -> str:
  """Converts a ExportedProgram to a MLIR text."""
  return stablehlo.exported_program_to_stablehlo(
      exported_program
  ).get_stablehlo_text()


def merged_bundle_to_tfl_model(
    merged_bundle: MergedBundle,
    signatures: list[signature_module.Signature],
    *,
    quant_config: Optional[qcfg.QuantConfig] = None,
    _tfl_converter_flags: dict = {},
) -> None:
  """Converts a StableHLOGraphModule to a tflite model.

  Args: shlo_bundle - model to export and save

    signatures: List of signatures from which names of the signatures is
    extracted.
    quant_config: User-defined quantization method and scheme of the model.
    _tfl_converter_flags: A nested dictionary allowing setting flags for the
    underlying tflite converter.
  """

  tf_module = tf.Module()

  shlo_bundle = merged_bundle.bundle

  shlo_bundle.additional_constants = [
      tf.Variable(v, trainable=False) for v in shlo_bundle.additional_constants
  ]
  tf_signatures: list[list[tf.TensorSpec]] = list(
      _make_tf_signature(func.meta, sig)
      for func, sig in zip(shlo_bundle.stablehlo_funcs, signatures)
  )

  tf_functions = [
      _wrap_as_tf_func(func, shlo_bundle, ep)
      for func, ep in zip(
          shlo_bundle.stablehlo_funcs, merged_bundle.exported_programs
      )
  ]

  tf_module.f = []
  for tf_sig, func in zip(tf_signatures, tf_functions):
    tf_module.f.append(
        tf.function(
            func,
            input_signature=tf_sig,
        )
    )

  tf_module._variables = (
      merged_bundle.deduped_tf_vars + shlo_bundle.additional_constants
  )
  del shlo_bundle
  gc.collect()

  tf_concrete_funcs = [
      func.get_concrete_function(*tf_sig)
      for func, tf_sig in zip(tf_module.f, tf_signatures)
  ]

  # We need to temporarily save since TFLite's from_concrete_functions does not
  # allow providing names for each of the concrete functions.

  with tempfile.TemporaryDirectory() as temp_dir_path:
    tf.saved_model.save(
        tf_module,
        temp_dir_path,
        signatures={
            sig.name: tf_concrete_funcs[idx]
            for idx, sig in enumerate(signatures)
        },
    )
    # Clean up intermediate memory early.
    del tf_functions
    del tf_module
    del tf_concrete_funcs
    gc.collect()

    converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir_path)
    converter._set_original_model_type(conversion_metadata_fb.ModelType.PYTORCH)
    converter._experimental_enable_composite_direct_lowering = True

    conversion_utils.set_tfl_converter_quant_flags(converter, quant_config)
    if (
        quant_config is not None
        and quant_config._quantizer_mode
        == quant_config._QuantizerMode.AI_EDGE_QUANTIZER
    ):
      translated_recipe = translate_recipe.translate_to_ai_edge_recipe(
          quant_config.generative_recipe
      )

    conversion_utils.apply_tfl_converter_flags(converter, _tfl_converter_flags)

    tflite_model = converter.convert()
    del converter
    gc.collect()

    if (
        quant_config is not None
        and quant_config._quantizer_mode
        == quant_config._QuantizerMode.AI_EDGE_QUANTIZER
    ):
      tflite_model = translate_recipe.quantize_model(
          tflite_model, translated_recipe
      )

  return tflite_model
