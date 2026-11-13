# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Compilation of models into models with EPContext nodes."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from .capi import _C
from ._options import SessionOptions
from ._types import GraphOptimizationLevel, OrtCompileApiFlags
from ._value import OrtValue


@dataclass
class OrtExternalInitializerInfo:
    """Location of the data of an initializer in an external file."""

    filepath: str
    file_offset: int
    byte_size: int


GetInitializerLocationFunc = Callable[
    [str, OrtValue, OrtExternalInitializerInfo | None], OrtExternalInitializerInfo | None
]


class ModelCompiler:
    """Compiler of ONNX models into models with EPContext nodes for the execution providers of the session options.

    See https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html.

    Parameters
    ----------
    sess_options
        Session options with the execution providers to compile for; see `SessionOptions.add_provider` and
        `SessionOptions.set_provider_selection_policy`.
    input_model_path_or_bytes
        Path of the model file or the serialized model.
    embed_compiled_data_into_model
        Whether the EPContext nodes embed the compiled data.
    external_initializers_file_path
        File for the initializers of the nodes that are not compiled; they stay in the model if None.
    external_initializers_size_threshold
        Size in bytes from which initializers are stored in `external_initializers_file_path`.
    flags
        Bitwise OR of `OrtCompileApiFlags`.
    graph_optimization_level
        Optimization level of the graph before compilation.
    get_initializer_location_func
        Called as ``f(name, value, external_info)`` for each initializer; returns the `OrtExternalInitializerInfo` of
        the file to which it wrote the initializer, or None to keep it in the model. `value` and `external_info` are
        only valid during the call.
    """

    def __init__(
        self,
        sess_options: SessionOptions,
        input_model_path_or_bytes: str | os.PathLike | bytes,
        embed_compiled_data_into_model: bool = False,
        external_initializers_file_path: str | os.PathLike | None = None,
        external_initializers_size_threshold: int = 1024,
        flags: int = OrtCompileApiFlags.NONE,
        graph_optimization_level: GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL,
        get_initializer_location_func: GetInitializerLocationFunc | None = None,
    ):
        if isinstance(input_model_path_or_bytes, (str, os.PathLike)):
            model: str | bytes = os.fspath(input_model_path_or_bytes)
            if not model:
                raise ValueError("Input model path is empty")
        elif isinstance(input_model_path_or_bytes, bytes):
            if not input_model_path_or_bytes:
                raise ValueError("Input model bytes array is empty")
            model = input_model_path_or_bytes
        else:
            raise TypeError(f"Unable to load from type '{type(input_model_path_or_bytes)}'")
        if external_initializers_file_path and not isinstance(external_initializers_file_path, (str, os.PathLike)):
            raise TypeError(
                f"Output external initializer filepath is of unexpected type '{type(external_initializers_file_path)}'"
            )
        if get_initializer_location_func is not None and external_initializers_file_path:
            raise ValueError(
                "Cannot initialize ModelCompiler with both `external_initializers_file_path` "
                "and `get_initializer_location_func`"
            )
        location_func = None
        if get_initializer_location_func is not None:

            def location_func(name: str, value: _C.OrtValue, info: tuple | None) -> tuple | None:
                external_info = None if info is None else OrtExternalInitializerInfo(*info)
                result = get_initializer_location_func(name, OrtValue(value), external_info)
                return None if result is None else (result.filepath, result.file_offset, result.byte_size)

        self._compiler = _C.ModelCompiler(
            sess_options._native_for_session(None),
            model,
            embed_compiled_data_into_model,
            os.fspath(external_initializers_file_path or ""),
            external_initializers_size_threshold,
            int(flags),
            int(graph_optimization_level),
            location_func,
        )

    def compile_to_file(self, output_model_path: str | os.PathLike | None = None) -> None:
        """Write the compiled model to `output_model_path`; to '<input>_ctx.onnx' if None."""
        if output_model_path is not None and not isinstance(output_model_path, (str, os.PathLike)):
            raise TypeError(f"Output model's filepath is of unexpected type '{type(output_model_path)}'")
        self._compiler.compile_to_file(os.fspath(output_model_path) if output_model_path else None)

    def compile_to_bytes(self) -> bytes:
        """Return the serialized compiled model."""
        return self._compiler.compile_to_bytes()

    def compile_to_stream(self, write_function: Callable[[bytes], None]) -> None:
        """Pass the serialized compiled model in chunks to `write_function`."""
        self._compiler.compile_to_stream(write_function)

    def set_ep_context_data_write_func(self, write_func: Callable[[str, _C.OrtEpContextData], None]) -> None:
        """Let ``write_func(name, data)`` store the external EPContext data during compilation.

        `data` is only valid during the call; read it with ``data.read(offset, length)``. Calls may be concurrent.
        """
        self._compiler.set_ep_context_data_write_func(write_func)

    def clear_ep_context_data_write_func(self) -> None:
        """Remove the callback set with `set_ep_context_data_write_func`."""
        self._compiler.set_ep_context_data_write_func(None)

    def set_ep_context_binary_information(self, output_directory: str | os.PathLike, model_name: str) -> None:
        """Set the directory and model name that execution providers use for their EPContext binary files."""
        self._compiler.set_ep_context_binary_information(os.fspath(output_directory), model_name)
