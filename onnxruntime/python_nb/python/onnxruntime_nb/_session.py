# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Inference sessions and their bindings of inputs and outputs."""

from __future__ import annotations

import collections.abc
import json
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _C
from ._exceptions import EPFail
from ._options import RunOptions, SessionOptions
from ._types import ONNX_TYPE_TENSOR, STRING, element_type, type_string
from ._value import OrtDevice, OrtValue, SparseTensor, from_python, tensor_from_array, to_python


@dataclass(frozen=True)
class NodeArg:
    """Name, type, and shape of an input or output of a model.

    The shape holds an int for each dimension of known size, a str for each named dimension, and None otherwise.
    """

    name: str
    type: str
    shape: list[int | str | None]

    def __str__(self) -> str:
        shape = ", ".join("None" if d is None else repr(d) for d in self.shape)
        return f"NodeArg(name='{self.name}', type='{self.type}', shape=[{shape}])"


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata of a model."""

    producer_name: str
    graph_name: str
    domain: str
    description: str
    graph_description: str
    version: int
    custom_metadata_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OrtEpAssignedNode:
    """A node that is assigned to an execution provider."""

    name: str
    domain: str
    op_type: str


@dataclass(frozen=True)
class OrtEpAssignedSubgraph:
    """A subgraph that is assigned to an execution provider."""

    ep_name: str
    nodes: tuple[OrtEpAssignedNode, ...]

    def get_nodes(self) -> list[OrtEpAssignedNode]:
        return list(self.nodes)


def _node_args(descriptions: list[tuple[str, tuple]]) -> list[NodeArg]:
    args = []
    for name, description in descriptions:
        shape = description[2] if description[0] in ("tensor", "sparse_tensor") else None
        args.append(NodeArg(name, type_string(description), list(shape or [])))
    return args


def check_and_normalize_provider_args(
    providers: Sequence[str | tuple[str, Mapping[Any, Any]]] | None,
    provider_options: Sequence[Mapping[Any, Any]] | None,
    available_provider_names: Sequence[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Return the names and string-valued options of `providers`, warning about unavailable or duplicate ones."""
    if providers is None:
        return [], []
    if not isinstance(providers, collections.abc.Sequence):
        raise ValueError("'providers' should be a sequence.")
    if provider_options is not None:
        if not isinstance(provider_options, collections.abc.Sequence):
            raise ValueError("'provider_options' should be a sequence.")
        if len(providers) != len(provider_options):
            raise ValueError("'providers' and 'provider_options' should be the same length if both are given.")
        if not all(isinstance(p, str) for p in providers):
            raise ValueError("Only string values for 'providers' are supported if 'provider_options' is given.")
        if not all(isinstance(o, dict) for o in provider_options):
            raise ValueError("'provider_options' values must be dicts.")
        pairs = list(zip(providers, provider_options, strict=True))
    else:
        pairs = []
        for provider in providers:
            if isinstance(provider, str):
                pairs.append((provider, {}))
            elif (
                isinstance(provider, tuple)
                and len(provider) == 2
                and isinstance(provider[0], str)
                and isinstance(provider[1], dict)
            ):
                pairs.append(provider)
            else:
                raise ValueError("'providers' values must be either strings or (string, dict) tuples.")
    normalized: dict[str, dict[str, str]] = {}
    for name, options in pairs:
        if name not in available_provider_names:
            warnings.warn(
                f"Specified provider '{name}' is not in available provider names."
                f"Available providers: '{', '.join(available_provider_names)}'",
                stacklevel=3,
            )
        if name in normalized:
            warnings.warn(f"Duplicate provider '{name}' encountered, ignoring.", stacklevel=3)
            continue
        normalized[name] = {str(k): str(v) for k, v in options.items()}
    return list(normalized), list(normalized.values())


class InferenceSession:
    """A session that runs an ONNX or ORT format model.

    Parameters
    ----------
    path_or_bytes
        Path of the model file or the serialized model.
    sess_options
        Session configuration; the defaults of `SessionOptions` if None.
    providers
        Execution providers in order of decreasing precedence, given as names or (name, options) tuples. If None, the
        providers of `sess_options` are used, or the CPU if it has none.
    provider_options
        Options for each entry of `providers`; only allowed if `providers` contains only names.
    **kwargs
        ``enable_fallback=0`` disables retrying with the CPU execution provider after failures of other execution
        providers; ``disabled_optimizers`` names graph optimizers to disable.
    """

    def __init__(
        self,
        path_or_bytes: str | bytes | os.PathLike,
        sess_options: SessionOptions | None = None,
        providers: Sequence[str | tuple[str, Mapping[Any, Any]]] | None = None,
        provider_options: Sequence[Mapping[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(path_or_bytes, (str, os.PathLike)):
            self._model: str | bytes = os.fspath(path_or_bytes)
        elif isinstance(path_or_bytes, bytes):
            self._model = path_or_bytes
        else:
            raise TypeError(f"Unable to load from type '{type(path_or_bytes)}'")
        self._sess_options = sess_options
        self._enable_fallback = int(kwargs.get("enable_fallback", 1)) == 1
        self._disabled_optimizers = sorted(kwargs.get("disabled_optimizers") or [])
        self._fallback_providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in self._provider_names(providers) and any(
            name in self._provider_names(providers)
            for name in ("TensorrtExecutionProvider", "NvTensorRTRTXExecutionProvider")
        ):
            self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._create_session(providers, provider_options)
        except (ValueError, RuntimeError) as e:
            if not self._enable_fallback or self._provider_names(providers) in ([], self._fallback_providers):
                raise
            warnings.warn(
                f"EP Error {e} when using {providers}. Falling back to {self._fallback_providers}.", stacklevel=2
            )
            self._create_session(self._fallback_providers, None)
            self._enable_fallback = False

    @staticmethod
    def _provider_names(providers: Sequence[Any] | None) -> list[str]:
        return [p[0] if isinstance(p, tuple) else p for p in providers or []]

    def _create_session(self, providers: Any, provider_options: Any) -> None:
        names, options = check_and_normalize_provider_args(providers, provider_options, _C.get_available_providers())
        sess_options = self._sess_options or SessionOptions()
        if names and sess_options.has_providers():
            warnings.warn(
                "Specified 'providers'/'provider_options' when creating InferenceSession but SessionOptions has "
                "already been configured with providers. InferenceSession will only use the providers "
                "passed to InferenceSession().",
                stacklevel=3,
            )
        if sess_options.use_per_session_threads and _C.has_global_thread_pools():
            raise RuntimeError("use_per_session_threads must be false when using a global thread pool")
        native_options = sess_options._native_for_session(list(zip(names, options, strict=True)) if names else None)
        if self._disabled_optimizers:
            native_options.add_config_entry(
                "optimization.disable_specified_optimizers", ",".join(self._disabled_optimizers)
            )
        self._sess = _C.Session(self._model, native_options)
        self._session_options = SessionOptions._from_native(self._sess.session_options(), sess_options)
        self._inputs = _node_args(self._sess.inputs)
        self._outputs = _node_args(self._sess.outputs)
        self._overridable_initializers = _node_args(self._sess.overridable_initializers)
        self._types = dict(self._sess.overridable_initializers + self._sess.inputs)
        self._providers = self._sess.provider_names
        self._provider_options = {name: self._sess.provider_options(i) for i, name in enumerate(self._providers)}

    def get_session_options(self) -> SessionOptions:
        """Return the options that the session uses, without execution providers.

        They include changes that ONNX Runtime made while creating the session, e.g. options loaded from the model.
        """
        return self._session_options

    def get_inputs(self) -> list[NodeArg]:
        return self._inputs

    def get_outputs(self) -> list[NodeArg]:
        return self._outputs

    def get_overridable_initializers(self) -> list[NodeArg]:
        """Return the initializers that inputs may override."""
        return self._overridable_initializers

    def get_modelmeta(self) -> ModelMetadata:
        return ModelMetadata(**self._sess.metadata)

    def get_input_memory_infos(self) -> list[_C.OrtMemoryInfo]:
        return self._sess.input_memory_infos

    def get_output_memory_infos(self) -> list[_C.OrtMemoryInfo]:
        return self._sess.output_memory_infos

    def get_input_epdevices(self) -> list[_C.OrtEpDevice | None]:
        return self._sess.input_ep_devices

    def get_providers(self) -> list[str]:
        """Return the registered execution providers in order of decreasing precedence."""
        return self._providers

    def get_provider_options(self) -> dict[str, dict[str, str]]:
        """Return the options of each registered execution provider."""
        return self._provider_options

    def get_provider_graph_assignment_info(self) -> list[OrtEpAssignedSubgraph]:
        """Return the subgraphs that each execution provider runs.

        Requires the session configuration entry 'session.record_ep_graph_assignment_info' to be '1'.
        """
        return [
            OrtEpAssignedSubgraph(ep_name, tuple(OrtEpAssignedNode(*node) for node in nodes))
            for ep_name, nodes in self._sess.graph_assignment()
        ]

    def set_providers(
        self,
        providers: Sequence[str | tuple[str, Mapping[Any, Any]]] | None = None,
        provider_options: Sequence[Mapping[Any, Any]] | None = None,
    ) -> None:
        """Re-create the session with other execution providers."""
        self._create_session(providers, provider_options)

    def disable_fallback(self) -> None:
        """Disable retrying with the CPU execution provider after failures of other execution providers."""
        self._enable_fallback = False

    def enable_fallback(self) -> None:
        """Enable retrying with the CPU execution provider after failures of other execution providers."""
        self._enable_fallback = True

    def _validate_input(self, feed_input_names: Sequence[str]) -> None:
        missing = [i.name for i in self._inputs if i.name not in feed_input_names and not i.type.startswith("optional")]
        if missing:
            raise ValueError(f"Required inputs ({missing}) are missing from input feed ({list(feed_input_names)}).")

    def _to_values(self, input_feed: Mapping[str, Any]) -> list[_C.OrtValue]:
        values = []
        for name, obj in input_feed.items():
            if obj is None:
                continue
            description = self._types.get(name)
            values.append(from_python(obj, description) if description else tensor_from_array(obj))
        return values

    def _run_with_fallback(self, run: Callable[[], Any]) -> Any:
        try:
            return run()
        except EPFail as e:
            if not self._enable_fallback:
                raise
            warnings.warn(
                f"EP Error: {e} using {self._providers}. Falling back to {self._fallback_providers}.", stacklevel=3
            )
            self.set_providers(self._fallback_providers)
            self._enable_fallback = False
            return run()

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any], run_options: RunOptions | None = None
    ) -> list[Any]:
        """Run the model and return the requested outputs; all outputs if `output_names` is None or empty.

        Tensors are returned as numpy arrays, sequences as lists, maps as dicts, and sparse tensors as `SparseTensor`.
        """
        self._validate_input(list(input_feed))
        self._validate_ownership(input_feed.values())
        names = list(output_names or [o.name for o in self._outputs])

        def run() -> list[Any]:
            values = self._to_values(input_feed)
            outputs = self._sess.run(_fed_names(input_feed), values, names, _native(run_options))
            # Outputs that are inputs share their buffers; copies keep them independent of the inputs.
            input_buffers = {v.data_ptr for v in values if _is_numeric_tensor(v)}
            return [
                to_python(o).copy() if _is_numeric_tensor(o) and o.data_ptr in input_buffers else to_python(o)
                for o in outputs
            ]

        return self._run_with_fallback(run)

    def run_async(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
        callback: Callable[[list[Any], Any, str], None],
        user_data: Any,
        run_options: RunOptions | None = None,
    ) -> None:
        """Run the model on a thread of ONNX Runtime and call ``callback(outputs, user_data, error)`` when done.

        `error` is an empty string on success.
        """
        self._validate_input(list(input_feed))
        names = list(output_names or [o.name for o in self._outputs])

        def done(outputs: list[_C.OrtValue], error: str) -> None:
            callback([to_python(o) for o in outputs], user_data, error)

        self._sess.run_async(_fed_names(input_feed), self._to_values(input_feed), names, _native(run_options), done)

    def run_with_ort_values(
        self,
        output_names: Sequence[str] | None,
        input_dict_ort_values: Mapping[str, OrtValue],
        run_options: RunOptions | None = None,
    ) -> list[OrtValue]:
        """Run the model with `OrtValue` inputs and return `OrtValue` outputs."""
        self._validate_input(list(input_dict_ort_values))
        self._validate_ownership(input_dict_ort_values.values())
        names = list(output_names or [o.name for o in self._outputs])

        def run() -> list[OrtValue]:
            values = [v._value for v in input_dict_ort_values.values()]
            outputs = self._sess.run(list(input_dict_ort_values), values, names, _native(run_options))
            return [OrtValue(o) for o in outputs]

        return self._run_with_fallback(run)

    def _validate_ownership(self, values: Any) -> None:
        for value in values:
            if isinstance(value, OrtValue) and value._session is not None and value._session is not self._sess:
                raise ValueError("Session-scoped OrtValue must be used with the session that created it.")

    def end_profiling(self) -> str:
        """End profiling and return the path of the profile file."""
        return self._sess.end_profiling()

    def get_profiling_start_time_ns(self) -> int:
        """Return the start time of profiling in nanoseconds, comparable to `time.monotonic_ns`."""
        return self._sess.profiling_start_time_ns

    def io_binding(self) -> IOBinding:
        return IOBinding(self)

    def run_with_iobinding(self, iobinding: IOBinding, run_options: RunOptions | None = None) -> None:
        """Run the model with the inputs and outputs bound by `iobinding`."""
        if iobinding._session is not self._sess:
            raise ValueError("IOBinding must be used with the session that created it.")
        self._sess.run_with_binding(iobinding._binding, _native(run_options))

    def create_ortvalue_from_shape_and_type(
        self, shape: Sequence[int], element_type: Any, device_type: str, device_id: int = 0, vendor_id: int = -1
    ) -> OrtValue:
        """Return an uninitialized tensor allocated by an allocator of this session.

        The tensor may only be used with this session.
        """
        memory_info = OrtDevice.make(device_type, device_id, vendor_id)._memory_info()
        value = self._sess.allocate(list(shape), _element_type(element_type), memory_info)
        return OrtValue(value, session=self._sess)

    def release_captured_graph(self, graph_annotation_id: int = 0) -> None:
        """Release the graph captured for the run option 'gpu_graph_id' `graph_annotation_id`."""
        self._sess.release_captured_graph(graph_annotation_id)

    def set_ep_dynamic_options(self, options: Mapping[str, str]) -> None:
        """Pass options to the execution providers that change their behavior at run time."""
        self._sess.set_ep_dynamic_options({str(k): str(v) for k, v in options.items()})

    def get_tuning_results(self) -> list[dict[str, Any]]:
        """Return the TunableOp tuning results, one dict with keys 'ep', 'results', and 'validators' per provider."""
        return json.loads(self._sess.tuning_results_json())

    def set_tuning_results(self, results: Sequence[Mapping[str, Any]], *, error_on_invalid: bool = False) -> None:
        """Load TunableOp tuning results in the format of `get_tuning_results`.

        Invalid results raise if `error_on_invalid` is True and are skipped with a warning otherwise.
        """
        self._sess.set_tuning_results_json(json.dumps(list(results)), error_on_invalid)


def _is_numeric_tensor(value: _C.OrtValue) -> bool:
    return value.has_value and value.onnx_type == ONNX_TYPE_TENSOR and value.element_type != STRING


def _fed_names(input_feed: Mapping[str, Any]) -> list[str]:
    """Return the names of the inputs that are not None; omitted optional inputs are None."""
    return [name for name, obj in input_feed.items() if obj is not None]


def _native(run_options: RunOptions | None) -> _C.RunOptions | None:
    return None if run_options is None else run_options._native


def _element_type(dtype: Any) -> int:
    return element_type(dtype)


class IOBinding:
    """Inputs and outputs of an `InferenceSession` bound to buffers, e.g. on a GPU, for `run_with_iobinding`."""

    def __init__(self, session: InferenceSession):
        self._session = session._sess
        self._binding = _C.IOBinding(session._sess)
        # Values whose buffers the binding uses.
        self._inputs: dict[str, Any] = {}
        self._outputs: dict[str, Any] = {}

    def bind_cpu_input(self, name: str, arr_on_cpu: np.ndarray) -> None:
        """Bind the input `name` to a numpy array, which must not change until the run."""
        value = tensor_from_array(arr_on_cpu)
        self._binding.bind_input(name, value)
        self._inputs[name] = value

    def bind_input(
        self, name: str, device_type: str, device_id: int, element_type: Any, shape: Sequence[int], buffer_ptr: int
    ) -> None:
        """Bind the input `name` to the buffer at address `buffer_ptr` on a device."""
        _check_not_string(element_type)
        value = _C.OrtValue.from_pointer(
            buffer_ptr, list(shape), _element_type(element_type), OrtDevice.make(device_type, device_id)._memory_info()
        )
        self._binding.bind_input(name, value)
        self._inputs[name] = value

    def bind_ortvalue_input(self, name: str, ortvalue: OrtValue) -> None:
        self._check_session(ortvalue)
        self._binding.bind_input(name, ortvalue._value)
        self._inputs[name] = ortvalue

    def bind_output(
        self,
        name: str,
        device_type: str = "cpu",
        device_id: int = 0,
        element_type: Any = None,
        shape: Sequence[int] | None = None,
        buffer_ptr: int | None = None,
    ) -> None:
        """Bind the output `name` to a device, or to the buffer at address `buffer_ptr` on the device."""
        memory_info = OrtDevice.make(device_type, device_id)._memory_info()
        if buffer_ptr is None:
            self._binding.bind_output_to_device(name, memory_info)
            self._outputs[name] = None
            return
        if element_type is None or shape is None:
            raise ValueError("`element_type` and `shape` are to be provided if pre-allocated memory is provided")
        _check_not_string(element_type)
        value = _C.OrtValue.from_pointer(buffer_ptr, list(shape), _element_type(element_type), memory_info)
        self._binding.bind_output(name, value)
        self._outputs[name] = value

    def bind_ortvalue_output(self, name: str, ortvalue: OrtValue) -> None:
        self._check_session(ortvalue)
        self._binding.bind_output(name, ortvalue._value)
        self._outputs[name] = ortvalue

    def _check_session(self, ortvalue: OrtValue) -> None:
        if ortvalue._session is not None and ortvalue._session is not self._session:
            raise ValueError("Session-scoped OrtValue must be bound to the session that created it.")

    def synchronize_inputs(self) -> None:
        self._binding.synchronize_inputs()

    def synchronize_outputs(self) -> None:
        self._binding.synchronize_outputs()

    def get_outputs(self) -> list[OrtValue]:
        """Return the outputs of the last run, which may reside on a device."""
        return [OrtValue(v) for v in self._binding.outputs()]

    def copy_outputs_to_cpu(self) -> list[Any]:
        """Return the outputs of the last run as numpy arrays, lists, or dicts on the CPU."""
        return [to_python(_to_cpu(value)) for value in self._binding.outputs()]

    def clear_binding_inputs(self) -> None:
        self._binding.clear_inputs()
        self._inputs.clear()

    def clear_binding_outputs(self) -> None:
        self._binding.clear_outputs()
        self._outputs.clear()


def _to_cpu(value: _C.OrtValue) -> _C.OrtValue:
    """Return `value`, or a copy on the CPU if it is a tensor on another device."""
    if value.onnx_type != ONNX_TYPE_TENSOR or value.memory_info.device_type == 0:
        return value
    cpu_value = _C.OrtValue.empty(value.shape, value.element_type, None)
    cpu_value.copy_from(value)
    return cpu_value


def _check_not_string(element_type: Any) -> None:
    if _element_type(element_type) == STRING:
        raise RuntimeError("Only binding non-string Tensors via raw pointers is supported")


def copy_tensors(src: Sequence[OrtValue], dst: Sequence[OrtValue], stream: Any = None) -> None:
    """Copy the data of the tensors `src` into the tensors `dst`, optionally on the `OrtSyncStream` `stream`."""
    _C.copy_tensors([s._value for s in src], [d._value for d in dst], stream)


__all__ = [
    "IOBinding",
    "InferenceSession",
    "ModelMetadata",
    "NodeArg",
    "OrtEpAssignedNode",
    "OrtEpAssignedSubgraph",
    "SparseTensor",
    "check_and_normalize_provider_args",
    "copy_tensors",
]
