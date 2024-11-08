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

"""Represents an ai_edge_torch model.

PyTorch models can be converted to this representation through
`ai_edge_torch.convert`.
"""
from __future__ import annotations

import abc
import re
from typing import Callable

import numpy.typing as npt
import tensorflow as tf

from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import

DEFAULT_SIGNATURE_NAME = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY


class Model(abc.ABC):
  """Represents and edge model."""

  @abc.abstractmethod
  def __call__(
      self,
      *args: npt.ArrayLike,
      signature_name: str = DEFAULT_SIGNATURE_NAME,
      **kwargs,
  ) -> npt.ArrayLike | tuple[npt.ArrayLike]:
    raise NotImplementedError()

  @abc.abstractmethod
  def export(self, path: str):
    raise NotImplementedError()

  @staticmethod
  def load(path: str) -> TfLiteModel:
    tflite_model = TfLiteModel.load(path)
    if tflite_model:
      return tflite_model

    raise ValueError(f'File format in {path} cannot be deserialized.')


class TfLiteModel(Model):
  """An edge model which uses tflite under-the-hood."""

  def __init__(self, tflite_model):
    """Initializes the TfLiteModel instance using a TFLite serialized object.

    Args:
      tflite_model: A TFlite serialized object.
    """
    self._tflite_model = tflite_model
    self._interpreter_builder = lambda: tfl_interpreter.Interpreter(
        model_content=self._tflite_model,
        experimental_default_delegate_latest_features=True,
    )

  def tflite_model(self) -> bytes:
    """Returns the wrapped tflite model."""
    return self._tflite_model

  def set_interpreter_builder(
      self, builder: Callable[[], tfl_interpreter.Interpreter]
  ) -> None:
    """Sets a custom interpreter builder.

    Args:
      builder: A function that returns a `tfl_interpreter.Interpreter` or its
        subclass.
    """
    self._interpreter_builder = builder

  def __call__(
      self,
      *args: npt.ArrayLike,
      signature_name: str = DEFAULT_SIGNATURE_NAME,
      **kwargs,
  ) -> npt.ArrayLike | tuple[npt.ArrayLike]:
    """Runs inference on the edge model using the provided arguments.

    Args:
      *args: The arguments to be passed to the model for inference.
      **kwargs: The arguments with specific names to be passed to the model for
        inference.
      signature_name: The name of the signature to be used for inference. The
        default signature is used if not provided.
    """
    interpreter = self._interpreter_builder()
    interpreter.allocate_tensors()

    signature_list = interpreter.get_signature_list()
    if signature_name not in signature_list:
      raise ValueError(
          'Invalid signature name provided. Available signatures:'
          f' {", ".join(signature_list.keys())}'
      )

    try:
      runner = interpreter.get_signature_runner(signature_name)
    except ValueError as exception:
      if 'Invalid signature_key provided.' in str(exception):
        raise ValueError(
            'Invalid signature key provided. Available signatures:'
            f' {list(signature_list.keys())}'
        )
      else:
        raise exception

    if len(signature_list[signature_name]['inputs']) != len(args) + len(kwargs):
      raise ValueError(
          'The model requires'
          f' {len(signature_list[signature_name]["inputs"])} arguments but'
          f' {len(args)} was provided.'
      )

    # Gather the input dictionary based on the signature.
    inputs = {f'args_{idx}': args[idx] for idx in range(len(args))}
    inputs = {**inputs, **kwargs}
    outputs = runner(**inputs)

    # When attempting to run a model, check if all the output tensors are named
    # output_<number>. If so, assume the pytorch model returned a tuple and not
    # a dictionary.
    output_heuristic = lambda key: bool(re.search(r'output_\d+', key))
    if all(output_heuristic(key) for key in outputs.keys()):
      return (
          outputs['output_0']
          if len(outputs) == 1
          else [outputs[f'output_{idx}'] for idx in range(len(outputs))]
      )

    return outputs

  def export(self, path: str) -> None:
    """Serializes the edge model to disk.

    Args:
      path: The path to file to which the model is serialized.
    """
    with open(path, 'wb') as file_handle:
      file_handle.write(self._tflite_model)

  @staticmethod
  def load(path: str) -> TfLiteModel | None:
    """Returns an edge (tflite) model by reading it from the disk.

    Args:
      str: The path to the model.
    """
    with open(path, 'rb') as file_handle:
      model_content = file_handle.read()

    # Check if this is indeed a tflite model:
    try:
      interpreter = tfl_interpreter.Interpreter(model_content=model_content)
      interpreter.get_signature_list()
    except:
      return None

    return TfLiteModel(model_content)
