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

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union

from ai_edge_torch import model
from ai_edge_torch._convert import conversion
from ai_edge_torch._convert import signature as signature_module
from ai_edge_torch.quantize import quant_config as qcfg
import torch


class Converter:
  """A converter for converting PyTorch models to edge models.

  This class allows adding multiple signatures to the converted edge model.
  """

  def __init__(self):
    self._signatures: list[signature_module.Signature] = []

  def signature(
      self,
      name: str,
      module: torch.nn.Module,
      sample_args=None,
      sample_kwargs=None,
      *,
      dynamic_shapes: Optional[Union[dict[str, Any], Tuple[Any, ...]]] = None,
  ) -> Converter:
    """Functions as an alias to `add_signature`."""
    return self.add_signature(
        name, module, sample_args, sample_kwargs, dynamic_shapes=dynamic_shapes
    )

  def add_signature(
      self,
      name: str,
      module: torch.nn.Module,
      sample_args=None,
      sample_kwargs=None,
      *,
      dynamic_shapes: Optional[Union[dict[str, Any], Tuple[Any, ...]]] = None,
  ) -> Converter:
    """Allows adding a new named torch model along with sample args to the conversion.

    Args:
      name: The name of the signature included in the converted edge model.
      module: The torch module to be converted.
      sample_args: Tuple of tensors by which the torch module will be traced
        with prior to conversion.
      sample_kwargs: Dict of str to tensor by which the torch module will be
        traced with prior to conversion.
      dynamic_shapes: Optional dict or tuple that specify dynamic shape
        specifications for each input in original order. See
        https://pytorch.org/docs/stable/export.html#expressing-dynamism for more
          details.

    Returns:
      The converter object itself.

    Raises:
      ValueError: If a signature with the provided name already exists.
    """

    if name in [sig.name for sig in self._signatures]:
      raise ValueError(
          f"A signature with the provided name ({name}) is already added."
      )

    if sample_args is None and sample_kwargs is None:
      raise ValueError("sample_args or sample_kwargs must be provided.")

    self._signatures.append(
        signature_module.Signature(
            name,
            module,
            sample_args,
            sample_kwargs,
            dynamic_shapes=dynamic_shapes,
        )
    )
    return self

  def convert(
      self,
      module: torch.nn.Module = None,
      sample_args=None,
      sample_kwargs=None,
      *,
      strict_export: Union[Literal["auto"], bool] = True,
      quant_config: Optional[qcfg.QuantConfig] = None,
      dynamic_shapes: Optional[Union[dict[str, Any], Tuple[Any, ...]]] = None,
      _ai_edge_converter_flags: Optional[dict[str, Any]] = None,
  ) -> model.TfLiteModel:
    """Finalizes the conversion and produces an edge model.

    This could be called with no arguments as follows:

      edge_model = Converter().signature(name, module, args).convert()

    Or it could be used to set the default signature for the converted edge
    model:

      edge_model =  Converter().convert(module, args)

    Args:
      module: The torch module to be converted.
      sample_args: Tuple of tensors by which the torch module will be traced
        with prior to conversion.
      sample_kwargs: Dict of str to tensor by which the torch module will be
        traced with prior to conversion.
      strict_export: Experimental `strict` arg for torch.export.export. When
        enabled, the export function will trace the program through TorchDynamo
        and ensure the soundness of the exported graph. When
        strict_export="auto", the function will try to export module in both
        modes and use the first one succeeds for downstream conversion.
      quant_config: User-defined quantization method and scheme of the model.
      dynamic_shapes: Optional dict or tuple that specify dynamic shape
        specifications for each input in original order. See
        https://pytorch.org/docs/stable/export.html#expressing-dynamism for more
          details.
      _ai_edge_converter_flags: A nested dictionary allowing setting flags for
        the underlying converter. This gives access to an implementation detail
        of this function and so needs to be treated as such. Please do not rely
        on this parameter except for local debugging as this can be removed in a
        future release.

    Returns:
      The converted edge model.

    Raises:
      ValueError: If the arguments are not provided as expected. See the example
      in this functions's comment.
    """
    if _ai_edge_converter_flags is None:
      _ai_edge_converter_flags = {}

    if module is not None:
      if (
          sample_args is not None or sample_kwargs is not None
      ):  # both module and args provided
        self.add_signature(
            model.DEFAULT_SIGNATURE_NAME,
            module,
            sample_args,
            sample_kwargs,
            dynamic_shapes=dynamic_shapes,
        )
      else:  # module is provided but not args
        raise ValueError(
            "sample_args or sample_kwargs must be provided if a module is"
            " specified."
        )
    return conversion.convert_signatures(
        self._signatures,
        strict_export=strict_export,
        quant_config=quant_config,
        _tfl_converter_flags=_ai_edge_converter_flags,
    )


def signature(
    name: str,
    module: torch.nn.Module,
    sample_args=None,
    sample_kwargs=None,
    dynamic_shapes: Optional[Union[dict[str, Any], Tuple[Any, ...]]] = None,
) -> Converter:
  """Initiates a Converter object with the provided signature.

  Args:
    name: The name of the signature included in the converted edge model.
    module: The torch module to be converted.
    sample_args: Tuple of tensors by which the torch module will be traced with
      prior to conversion.
    sample_kwargs: Dict of str to tensor by which the torch module will be
      traced with prior to conversion.
    dynamic_shapes: Optional dict or tuple that specify dynamic shape
      specifications for each input in original order. See
      https://pytorch.org/docs/stable/export.html#expressing-dynamism for more
        details.

  Returns:
    A Converter object with the provided signature.

  Example:
    converter = ai_edge_torch.signature(name, module, args)
    edge_model = converter.convert()
  """
  return Converter().signature(
      name, module, sample_args, sample_kwargs, dynamic_shapes=dynamic_shapes
  )


def convert(
    module: torch.nn.Module = None,
    sample_args=None,
    sample_kwargs=None,
    *,
    strict_export: Union[Literal["auto"], bool] = True,
    quant_config: Optional[qcfg.QuantConfig] = None,
    dynamic_shapes: Optional[Union[dict[str, Any], Tuple[Any, ...]]] = None,
    _ai_edge_converter_flags: Optional[dict[str, Any]] = None,
) -> model.TfLiteModel:
  """Converts a PyTorch model to an edge model with a default signature.

  Args:
    module: The torch module to be converted.
    sample_args: Tuple of tensors by which the torch module will be traced with
      prior to conversion.
    sample_kwargs: Dict of str to tensor by which the torch module will be
      traced with prior to conversion.
    strict_export: Experimental `strict` arg for torch.export.export. When
      enabled, the export function will trace the program through TorchDynamo
      and ensure the soundness of the exported graph. When strict_export="auto",
      the function will try to export module in both modes and use the first one
      succeeds for downstream conversion.
    quant_config: User-defined quantization method and scheme of the model.
    dynamic_shapes: Optional dict or tuple that specify dynamic shape
      specifications for each input in original order. See
      https://pytorch.org/docs/stable/export.html#expressing-dynamism for more
        details.
    _ai_edge_converter_flags: A nested dictionary allowing setting flags for the
      underlying converter. This gives access to an implementation detail of
      this function and so needs to be treated as such. Please do not rely on
      this parameter except for local debugging as this can be removed in a
      future release.

  Returns:
    The converted edge model.

  Example:
    edge_model = ai_edge_torch.convert(module, args)
  """

  if _ai_edge_converter_flags is None:
    _ai_edge_converter_flags = {}

  return Converter().convert(
      module,
      sample_args,
      sample_kwargs,
      strict_export=strict_export,
      quant_config=quant_config,
      dynamic_shapes=dynamic_shapes,
      _ai_edge_converter_flags=_ai_edge_converter_flags,
  )
