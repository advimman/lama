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
"""APIs to convert lowered MLIR from PyTorch to TensorFlow and TFLite artifacts."""

import re
import tempfile

import tensorflow as tf
import torch

from tensorflow.compiler.tf2xla.python import xla as tfxla

from . import export
from . import export_utils


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


def _mangle_tf_root_scope_name(name):
  r"""Build the mangled name for tf.Variable.

  TF has more restricted constrain on the variable names at root scope. Root
  scope name constrain: [A-Za-z0-9.][A-Za-z0-9_.\\-/]* Non-root scope name
  constrain: [A-Za-z0-9_.\\-/]*
  https://github.com/tensorflow/tensorflow/blob/51b601fa6bb7e801c0b6ae73c25580e40a8b5745/tensorflow/python/framework/ops.py#L3301-L3302
  The state_dict key doesn't have such constrain, the name need to be mangled
  when a root-scoped TF variable is created.

  FX Graph Node may contain characters other than [A-Za-z0-9_.\\-/], replace
  offending characters with '_'.

  Args:
    name: the tensor name to be mangled.

  Returns:
    Mangled name in str.
  """
  if name[0] in "._\\-/":
    name = "k" + name
  name = re.sub(r"[^^\w\-/\\]+", "_", name)
  return name


def _build_tf_state_dict(
    lowered: export.MlirLowered,
) -> dict[str, tf.Variable]:
  """Build a dictionary of tf.Variable from the state_dict in lowered."""
  tf_state_dict = {}
  for sig in lowered.input_signature:
    if sig.input_spec.is_parameter:
      name = sig.input_spec.name
      tf_state_dict[name] = tf.Variable(
          lowered.state_dict[name].detach().numpy(),
          trainable=False,
          name=_mangle_tf_root_scope_name(name),
      )
  return tf_state_dict


def _extract_call_args(
    lowered: export.MlirLowered,
    args,
    tf_state_dict: dict[str, tf.Variable],
):
  """Extract the flattened inputs to built tf.function."""
  call_args = []
  for sig in lowered.input_signature:
    if sig.input_spec.is_user_input:
      call_args.append(args[sig.input_spec.i])
    elif sig.input_spec.is_parameter:
      name = sig.input_spec.name
      call_args.append(tf_state_dict[name])
  return call_args


def _wrap_as_tf_func(lowered, tf_state_dict):
  """Build tf.function from lowered and tf_state_dict."""

  def inner(*args):
    t_outs = [torch_dtype_to_tf(sig.dtype) for sig in lowered.output_signature]
    s_outs = [_get_shape_with_dynamic(sig) for sig in lowered.output_signature]
    call_args = _extract_call_args(lowered, args, tf_state_dict)
    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=t_outs,  # dtype information
        Sout=s_outs,  # Shape information
        function_list=[],
        module=lowered.module_bytecode,
    )

  return inner


def _make_input_signatures(
    lowered: export.MlirLowered,
) -> list[tf.TensorSpec]:
  """Build the input signatures in tf.TensorSpec for building tf.function."""
  user_input_signature = sorted(
      [sig for sig in lowered.input_signature if sig.input_spec.is_user_input],
      key=lambda sig: sig.input_spec.i,
  )
  tf_signatures = []

  for sig in user_input_signature:
    shape = _get_shape_with_dynamic(sig)
    tf_signatures.append(
        tf.TensorSpec(
            shape=shape,
            dtype=torch_dtype_to_tf(sig.dtype),
            name=f"args_{sig.input_spec.i}",
        )
    )
  return tf_signatures


def mlir_to_tf_function(lowered: export.MlirLowered):
  """Convert the MLIR lowered to a executable tf.function."""
  tf_state_dict = _build_tf_state_dict(lowered)
  return tf.function(
      _wrap_as_tf_func(lowered, tf_state_dict),
      input_signature=_make_input_signatures(lowered),
  )


def mlir_to_flatbuffer(lowered: export.MlirLowered):
  """Convert the MLIR lowered to a TFLite flatbuffer binary."""
  tf_state_dict = _build_tf_state_dict(lowered)
  signature_names = [tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  tf_signatures = [_make_input_signatures(lowered)]
  tf_functions = [_wrap_as_tf_func(lowered, tf_state_dict)]

  tf_module = tf.Module()
  tf_module.f = []

  for tf_sig, func in zip(tf_signatures, tf_functions):
    tf_module.f.append(
        tf.function(
            func,
            input_signature=tf_sig,
        )
    )

    tf_module._variables = list(tf_state_dict.values())

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
            sig_name: tf_concrete_funcs[idx]
            for idx, sig_name in enumerate(signature_names)
        },
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir_path)
    tflite_model = converter.convert()

  return tflite_model
