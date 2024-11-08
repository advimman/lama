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

import dataclasses
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from ai_edge_torch import odml_torch
from ai_edge_torch._convert import conversion_utils
from ai_edge_torch._convert import signature as signature_module
from ai_edge_torch.lowertools import common_utils
from ai_edge_torch.lowertools import translate_recipe
from ai_edge_torch.odml_torch import export
from ai_edge_torch.odml_torch import export_utils
from ai_edge_torch.quantize import quant_config as qcfg
import tensorflow as tf
import torch

from tensorflow.compiler.tf2xla.python import xla as tfxla
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb

MlirBundle = odml_torch.export.MlirLowered


@dataclasses.dataclass
class MergedBundle:
  """A bundle of MlirLowered that has been merged."""

  bundles: list[odml_torch.export.MlirLowered]
  exported_programs: list[torch.export.ExportedProgram]
  deduped_tf_vars: list[tf.Variable]


def torch_dtype_to_tf(dtype):
  return {
      torch.double: tf.float64,
      torch.float32: tf.float32,
      torch.half: tf.float16,
      torch.long: tf.int64,
      torch.int32: tf.int32,
      torch.int16: tf.int16,
      torch.bool: tf.bool,
  }.get(dtype)


def _get_shape_with_dynamic(signature: export.VariableSignature):
  return [
      None if export_utils.is_torch_dynamic(s) else s for s in signature.shape
  ]


def _extract_call_args(
    bundle: export.MlirLowered,
    args: Tuple[Any],
    tf_state_dict: Dict[str, tf.Variable],
):
  call_args = []
  for sig in bundle.input_signature:
    if sig.input_spec.is_user_input:
      call_args.append(args[sig.input_spec.i])
    elif sig.input_spec.is_parameter:
      name = sig.input_spec.name
      call_args.append(tf_state_dict[name])
  return call_args


def _wrap_as_tf_func(
    bundle: export.MlirLowered,
    tf_state_dict: Dict[str, tf.Variable],
    exported_program: torch.export.ExportedProgram,
):
  def inner(*args):
    t_outs = [torch_dtype_to_tf(sig.dtype) for sig in bundle.output_signature]
    s_outs = [_get_shape_with_dynamic(sig) for sig in bundle.output_signature]
    call_args = _extract_call_args(bundle, args, tf_state_dict)
    # HACK: In OSS, we use MLIR pybinding and StableHLO dialect from JAX's
    # build, which may not have the same StableHLO version as what used in
    # TFLite converter. Therefore we always serialize MLIR module in VHLO.
    # TODO(b/362798610) Build MLIR pybinding in ai-edge-torch release.
    call_module_return = tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=t_outs,  # dtype information
        Sout=s_outs,  # Shape information
        function_list=[],
        module=bundle.module_bytecode_vhlo,
    )
    spec = exported_program.call_spec.out_spec

    # The module returning a flat array.
    if not spec.context:
      return call_module_return

    flat_names = common_utils.flat_dict_names(spec.children_specs, spec.context)
    return {name: value for name, value in zip(flat_names, call_module_return)}

  return inner


def _make_tf_signature(
    input_signature: list[export.VariableSignature],
    signature: signature_module.Signature,
) -> List[tf.TensorSpec]:
  input_names = signature.flat_arg_names
  user_input_signature = sorted(
      [sig for sig in input_signature if sig.input_spec.is_user_input],
      key=lambda sig: sig.input_spec.i,
  )
  tf_signature = []

  for sig in user_input_signature:
    shape = _get_shape_with_dynamic(sig)
    tf_signature.append(
        tf.TensorSpec(
            shape=shape,
            dtype=torch_dtype_to_tf(sig.dtype),
            name=input_names[sig.input_spec.i],
        )
    )
  return tf_signature


def merged_bundle_to_tfl_model(
    merged_bundle: MergedBundle,
    signatures: list[signature_module.Signature],
    *,
    quant_config: Optional[qcfg.QuantConfig] = None,
    _tfl_converter_flags: dict = {},
):
  tf_state_dict = merged_bundle.bundles[0].state_dict

  tf_signatures = [
      _make_tf_signature(bundle.input_signature, sig)
      for bundle, sig in zip(merged_bundle.bundles, signatures)
  ]
  tf_functions = [
      _wrap_as_tf_func(bundle, tf_state_dict, ep)
      for bundle, ep in zip(
          merged_bundle.bundles, merged_bundle.exported_programs
      )
  ]

  tf_module = tf.Module()
  tf_module.f = []

  for tf_sig, func in zip(tf_signatures, tf_functions):
    tf_module.f.append(
        tf.function(
            func,
            input_signature=tf_sig,
        )
    )

  tf_module._variables = merged_bundle.deduped_tf_vars

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

    converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir_path)
    converter._set_original_model_type(conversion_metadata_fb.ModelType.PYTORCH)
    converter._experimental_enable_composite_direct_lowering = True
    converter.model_origin_framework = "PYTORCH"

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

    if (
        quant_config is not None
        and quant_config._quantizer_mode
        == quant_config._QuantizerMode.AI_EDGE_QUANTIZER
    ):
      tflite_model = translate_recipe.quantize_model(
          tflite_model, translated_recipe
      )

  return tflite_model


def exported_program_to_mlir_text(
    exported_program: torch.export.ExportedProgram,
) -> str:
  """Converts a ExportedProgram to a MLIR text."""
  return odml_torch.export.exported_program_to_mlir(exported_program).get_text(
      enable_debug_info=True
  )


def exported_program_to_mlir(
    exported_program: torch.export.ExportedProgram,
    sample_args: tuple[torch.Tensor],
) -> export.MlirLowered:
  """Converts a ExportedProgram to a MlirLowered."""
  return odml_torch.export.exported_program_to_mlir(exported_program)


def merge_mlir_bundles(
    bundles: list[export.MlirLowered],
    signatures: list[signature_module.Signature],
    exported_programs: list[torch.export.ExportedProgram],
) -> MergedBundle:
  """Merges a list of MlirLowered into one."""
  state_dict, deduped_vars = common_utils.gather_state_dict(
      exported_programs, signatures
  )

  merged_bundle = MergedBundle(
      bundles=bundles.copy(),
      exported_programs=exported_programs,
      deduped_tf_vars=deduped_vars,
  )
  for bundle, signature in zip(merged_bundle.bundles, signatures):
    bundle.state_dict = state_dict

    for var_sig in bundle.input_signature:
      if var_sig.input_spec.is_parameter:
        var_sig.input_spec.name = signature.name + "_" + var_sig.input_spec.name

  return merged_bundle
