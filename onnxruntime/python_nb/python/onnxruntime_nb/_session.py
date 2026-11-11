# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
import enum
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import _C


class GraphOptimizationLevel(enum.IntEnum):
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_LAYOUT = 3
    ORT_ENABLE_ALL = 99


class ExecutionMode(enum.IntEnum):
    ORT_SEQUENTIAL = 0
    ORT_PARALLEL = 1


@dataclasses.dataclass
class SessionOptions:
    """Configuration of an `InferenceSession`.

    A `log_severity_level` of -1 uses the level of the environment.
    """

    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0
    graph_optimization_level: GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
    execution_mode: ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
    log_severity_level: int = -1
    log_verbosity_level: int = 0
    logid: str = ""
    enable_profiling: bool = False
    profile_file_prefix: str = "onnxruntime_profile_"
    optimized_model_filepath: str = ""
    enable_cpu_mem_arena: bool = True
    enable_mem_pattern: bool = True
    _config_entries: dict[str, str] = dataclasses.field(default_factory=dict, init=False, repr=False)

    def add_session_config_entry(self, key: str, value: str) -> None:
        """Set a session configuration entry; see onnxruntime_session_options_config_keys.h for the keys."""
        self._config_entries[key] = value

    def get_session_config_entry(self, key: str) -> str:
        """Return the value of a session configuration entry."""
        try:
            return self._config_entries[key]
        except KeyError:
            raise RuntimeError(f"SessionOptions does not have configuration with key: {key}") from None

    def _to_native(self) -> _C.SessionOptions:
        native = _C.SessionOptions()
        native.set_intra_op_num_threads(self.intra_op_num_threads)
        native.set_inter_op_num_threads(self.inter_op_num_threads)
        native.set_graph_optimization_level(GraphOptimizationLevel(self.graph_optimization_level))
        native.set_execution_mode(ExecutionMode(self.execution_mode))
        if self.log_severity_level >= 0:
            native.set_log_severity_level(self.log_severity_level)
        native.set_log_verbosity_level(self.log_verbosity_level)
        if self.logid:
            native.set_logid(self.logid)
        if self.enable_profiling:
            native.enable_profiling(self.profile_file_prefix)
        if self.optimized_model_filepath:
            native.set_optimized_model_filepath(self.optimized_model_filepath)
        native.set_cpu_mem_arena(self.enable_cpu_mem_arena)
        native.set_mem_pattern(self.enable_mem_pattern)
        for key, value in self._config_entries.items():
            native.add_config_entry(key, value)
        return native


def _normalize_providers(
    providers: Sequence[str | tuple[str, Mapping[Any, Any]]] | None,
    provider_options: Sequence[Mapping[Any, Any]] | None,
) -> list[tuple[str, dict[str, str]]]:
    """Return (name, options) pairs with string-valued options."""
    if providers is None:
        return []
    if provider_options is not None:
        if len(providers) != len(provider_options):
            raise ValueError("'providers' and 'provider_options' should be the same length if both are given.")
        if not all(isinstance(provider, str) for provider in providers):
            raise ValueError("Only string values for 'providers' are supported if 'provider_options' is given.")
        pairs = zip(providers, provider_options, strict=True)
    else:
        pairs = (provider if isinstance(provider, tuple) else (provider, {}) for provider in providers)
    return [(name, {str(k): str(v) for k, v in options.items()}) for name, options in pairs]


class InferenceSession:
    """Inference session for an ONNX model.

    Parameters
    ----------
    path
        Path of the model file.
    sess_options
        Session configuration; the defaults of `SessionOptions` if None.
    providers
        Execution providers in order of decreasing precedence, given as names or (name, options) tuples.
        Without providers, the session runs on the CPU.
    provider_options
        Options for each entry of `providers`; only allowed if `providers` contains only names.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        sess_options: SessionOptions | None = None,
        providers: Sequence[str | tuple[str, Mapping[Any, Any]]] | None = None,
        provider_options: Sequence[Mapping[Any, Any]] | None = None,
    ):
        native_options = (sess_options or SessionOptions())._to_native()
        for name, options in _normalize_providers(providers, provider_options):
            native_options.append_execution_provider(name, options)
        self._session = _C.Session(path, native_options)

    @property
    def input_names(self) -> list[str]:
        return self._session.input_names

    @property
    def output_names(self) -> list[str]:
        return self._session.output_names

    def run(self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]) -> list[np.ndarray]:
        """Run the model and return the requested outputs; all outputs if `output_names` is None or empty."""
        arrays = {}
        strings = {}
        for name, value in input_feed.items():
            array = np.asarray(value)
            if array.dtype.kind in "OUS":
                elements = [x.decode() if isinstance(x, bytes) else str(x) for x in array.ravel().tolist()]
                strings[name] = (elements, list(array.shape))
            else:
                arrays[name] = array if array.flags.c_contiguous else array.copy(order="C")
        outputs = self._session.run(list(output_names or self.output_names), arrays, strings)
        return [_string_array(*output) if isinstance(output, tuple) else output for output in outputs]


def _string_array(elements: list[str], shape: list[int]) -> np.ndarray:
    array = np.empty(len(elements), dtype=object)
    array[:] = elements
    return array.reshape(shape)
