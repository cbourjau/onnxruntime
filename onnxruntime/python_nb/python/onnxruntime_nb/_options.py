# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Options of sessions, runs, and arena allocators."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import _C
from ._types import ExecutionMode, ExecutionOrder, GraphOptimizationLevel, OrtExecutionProviderDevicePolicy
from ._value import OrtValue


class _Setting:
    """A setting of `SessionOptions` that is stored in the native options."""

    def __init__(self, apply: Callable[[SessionOptions, Any], None], kind: type):
        self._apply = apply
        self._kind = kind

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: SessionOptions | None, owner: type | None = None) -> Any:
        if obj is None:
            return self
        return self._kind(getattr(obj._native, self._name))

    def __set__(self, obj: SessionOptions, value: Any) -> None:
        self._apply(obj, self._kind(value))


class _UnsupportedSetting:
    """A setting of `SessionOptions` that the C API does not support; it is stored but has no effect."""

    def __init__(self, default: Any, kind: type):
        self._default = default
        self._kind = kind

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: SessionOptions | None, owner: type | None = None) -> Any:
        if obj is None:
            return self
        return obj._unsupported.get(self._name, self._default)

    def __set__(self, obj: SessionOptions, value: Any) -> None:
        obj._unsupported[self._name] = self._kind(value)


def _set_per_session_threads(o: SessionOptions, enable: bool) -> None:
    if not enable:
        o._native.disable_per_session_threads()
    elif not o.use_per_session_threads:
        raise ValueError("per-session threads cannot be re-enabled")


_DEFAULT_PROFILE_FILE_PREFIX = "onnxruntime_profile_"


class SessionOptions:
    """Configuration of an `InferenceSession`."""

    enable_cpu_mem_arena = _Setting(lambda o, v: o._native.set_cpu_mem_arena(v), bool)
    optimized_model_filepath = _Setting(lambda o, v: o._native.set_optimized_model_filepath(v), os.fspath)
    enable_mem_pattern = _Setting(lambda o, v: o._native.set_mem_pattern(v), bool)
    #: Not supported by the C API; setting it has no effect.
    enable_mem_reuse = _UnsupportedSetting(True, bool)
    logid = _Setting(lambda o, v: o._native.set_logid(v), str)
    log_severity_level = _Setting(lambda o, v: o._native.set_log_severity_level(v), int)
    log_verbosity_level = _Setting(lambda o, v: o._native.set_log_verbosity_level(v), int)
    use_per_session_threads = _Setting(_set_per_session_threads, bool)
    intra_op_num_threads = _Setting(lambda o, v: o._native.set_intra_op_num_threads(v), int)
    inter_op_num_threads = _Setting(lambda o, v: o._native.set_inter_op_num_threads(v), int)
    execution_mode = _Setting(lambda o, v: o._native.set_execution_mode(v), ExecutionMode)
    #: Not supported by the C API; setting it has no effect.
    execution_order = _UnsupportedSetting(ExecutionOrder.DEFAULT, ExecutionOrder)
    graph_optimization_level = _Setting(lambda o, v: o._native.set_graph_optimization_level(v), GraphOptimizationLevel)
    use_deterministic_compute = _Setting(lambda o, v: o._native.set_deterministic_compute(v), bool)
    enable_profiling = _Setting(lambda o, v: o._native.set_profiling(v, o._profile_file_prefix), bool)

    @property
    def profile_file_prefix(self) -> str:
        """Prefix of the file that profiling results are written to."""
        return self._native.profile_file_prefix if self.enable_profiling else self._profile_file_prefix

    @profile_file_prefix.setter
    def profile_file_prefix(self, value: str) -> None:
        # The native options store the prefix only while profiling is enabled.
        self._profile_file_prefix = str(value)
        if self.enable_profiling:
            self._native.set_profiling(True, self._profile_file_prefix)

    def __init__(self) -> None:
        self._native = _C.SessionOptions()
        self._unsupported: dict[str, Any] = {}
        self._profile_file_prefix = _DEFAULT_PROFILE_FILE_PREFIX
        # Execution providers as (name, options) or (devices, options), applied to the options of each session.
        self._providers: list[tuple[Any, dict[str, str]]] = []
        self._policy: OrtExecutionProviderDevicePolicy | Callable | None = None

    def add_provider(self, provider_name: str, provider_options: Mapping[str, Any] | None = None) -> None:
        """Add an execution provider by name."""
        options = _string_dict(provider_options)
        _C.SessionOptions().append_execution_provider(provider_name, options)  # validates the arguments
        self._providers.append((provider_name, options))

    def add_provider_for_devices(
        self, ep_devices: Sequence[_C.OrtEpDevice], provider_options: Mapping[str, Any] | None = None
    ) -> None:
        """Add the execution provider of `ep_devices`, which must all belong to the same execution provider."""
        options = _string_dict(provider_options)
        _C.SessionOptions().append_execution_provider_for_devices(list(ep_devices), options)
        self._providers.append((list(ep_devices), options))

    def set_provider_selection_policy(self, policy: OrtExecutionProviderDevicePolicy) -> None:
        """Let ONNX Runtime select the execution providers according to `policy`."""
        self._policy = OrtExecutionProviderDevicePolicy(policy)

    def set_provider_selection_policy_delegate(self, delegate: Callable) -> None:
        """Let `delegate` select the execution providers.

        Its signature is ``delegate(ep_devices, model_metadata, runtime_metadata, max_selections)``; it returns the
        selected `OrtEpDevice` instances.
        """
        if not callable(delegate):
            raise TypeError("the provider selection policy delegate must be callable")
        self._policy = delegate

    def has_providers(self) -> bool:
        """Return True if execution providers or a selection policy are configured."""
        return bool(self._providers) or self._policy is not None

    def add_session_config_entry(self, config_key: str, config_value: str) -> None:
        """Set a session configuration entry; see onnxruntime_session_options_config_keys.h for the keys."""
        self._native.add_config_entry(config_key, config_value)

    def get_session_config_entry(self, config_key: str) -> str:
        """Return the value of a session configuration entry."""
        value = self._native.get_config_entry(config_key)
        if value is None:
            raise RuntimeError(f"SessionOptions does not have configuration with key: {config_key}")
        return value

    def add_free_dimension_override_by_denotation(self, dim_denotation: str, dim_value: int) -> None:
        """Set the size of the free dimensions with the denotation `dim_denotation`."""
        self._native.add_free_dimension_override_by_denotation(dim_denotation, dim_value)

    def add_free_dimension_override_by_name(self, dim_name: str, dim_value: int) -> None:
        """Set the size of the free dimensions named `dim_name`."""
        self._native.add_free_dimension_override_by_name(dim_name, dim_value)

    def register_custom_ops_library(self, library_path: str | os.PathLike) -> None:
        """Load the custom operators of a shared library."""
        self._native.register_custom_ops_library(os.fspath(library_path))

    def add_initializer(self, name: str, value: OrtValue) -> None:
        """Override the initializer `name` with `value`, which must stay unchanged while sessions use it."""
        if not isinstance(value, OrtValue):
            raise TypeError("The provided Python object must be an OrtValue")
        self._native.add_initializer(name, value._value)

    def add_external_initializers(self, names: Sequence[str], ort_values: Sequence[OrtValue]) -> None:
        """Provide the data of initializers stored outside the model."""
        if len(names) != len(ort_values):
            raise ValueError("Expecting names and ort_values lists to have equal length")
        self._native.add_external_initializers([str(n) for n in names], [v._value for v in ort_values])

    def add_external_initializers_from_files_in_memory(
        self, names: Sequence[str], buffers: Sequence[Any], lengths: Sequence[int]
    ) -> None:
        """Provide the content of the external data files `names` of the model."""
        if not len(names) == len(buffers) == len(lengths):
            raise ValueError("expecting 'names', 'buffers' and 'lengths' to have equal length")
        data = [bytes(memoryview(b).cast("B")[:n]) for b, n in zip(buffers, lengths, strict=True)]
        self._native.add_external_initializers_from_files_in_memory(list(names), data)

    def set_ep_context_data_read_func(self, read_func: Callable, max_data_size: int) -> None:
        """Let `read_func(name, buffer)` supply external EPContext data while sessions load.

        It must call ``buffer.allocate(size)`` once with `size` at most `max_data_size` and then fill the buffer with
        ``buffer.write(data, offset)``. The buffer is only valid during the call.
        """
        if read_func is None:
            raise TypeError("EPContext data read callback must not be None.")
        self._native.set_ep_context_data_read_func(read_func, max_data_size)

    def clear_ep_context_data_read_func(self) -> None:
        """Remove the callback set with `set_ep_context_data_read_func`."""
        self._native.set_ep_context_data_read_func(None)

    def set_load_cancellation_flag(self, value: bool) -> None:
        """Request that sessions being created with these options cancel loading."""
        self._native.set_load_cancellation_flag(value)

    @classmethod
    def _from_native(cls, native: _C.SessionOptions, unsupported: SessionOptions) -> SessionOptions:
        """Return options without execution providers that wrap `native`, with the unsupported settings of
        `unsupported`."""
        options = cls.__new__(cls)
        options._native = native
        options._unsupported = dict(unsupported._unsupported)
        options._profile_file_prefix = (
            native.profile_file_prefix if native.enable_profiling else unsupported._profile_file_prefix
        )
        options._providers = []
        options._policy = None
        return options

    def _native_for_session(self, providers: Sequence[tuple[str, dict[str, str]]] | None) -> _C.SessionOptions:
        """Return native options with `providers`, or the providers of these options if None."""
        native = self._native.clone()
        if providers is not None:
            for name, options in providers:
                _append_provider(native, name, options)
            return native
        for provider, options in self._providers:
            if isinstance(provider, str):
                native.append_execution_provider(provider, options)
            else:
                native.append_execution_provider_for_devices(provider, options)
        if callable(self._policy):
            native.set_ep_selection_policy_delegate(self._policy)
        elif self._policy is not None:
            native.set_ep_selection_policy(self._policy)
        return native


def _append_provider(native: _C.SessionOptions, name: str, options: dict[str, str]) -> None:
    """Append the execution provider `name`; skip it if it is known but unavailable in this build."""
    if name in _C.get_available_providers():
        native.append_execution_provider(name, options)
        return
    devices = [d for d in _C.get_ep_devices() if d.ep_name == name]
    if devices:
        # Execution providers of registered libraries.
        native.append_execution_provider_for_devices(devices[:1], options)
    elif name not in _C.get_all_providers():
        raise RuntimeError(f"Unknown Provider Type: {name}")


def _string_dict(options: Mapping[Any, Any] | None) -> dict[str, str]:
    return {str(k): str(v) for k, v in (options or {}).items()}


class RunOptions:
    """Configuration of a single run of a session."""

    def __init__(self) -> None:
        self._native = _C.RunOptions()
        self._terminate = False
        self._adapters: list[Any] = []

    @property
    def log_severity_level(self) -> int:
        """Log severity level of the run: 0 (verbose), 1 (info), 2 (warning), 3 (error), or 4 (fatal)."""
        return self._native.log_severity_level

    @log_severity_level.setter
    def log_severity_level(self, value: int) -> None:
        self._native.log_severity_level = value

    @property
    def log_verbosity_level(self) -> int:
        """Verbosity of the run; applies to debug builds if `log_severity_level` is 0."""
        return self._native.log_verbosity_level

    @log_verbosity_level.setter
    def log_verbosity_level(self, value: int) -> None:
        self._native.log_verbosity_level = value

    @property
    def logid(self) -> str:
        """Tag of the log messages of the run."""
        return self._native.logid

    @logid.setter
    def logid(self, value: str) -> None:
        self._native.logid = value

    @property
    def terminate(self) -> bool:
        """Set to True to make the runs using these options fail as soon as possible."""
        return self._terminate

    @terminate.setter
    def terminate(self, value: bool) -> None:
        self._native.set_terminate(bool(value))
        self._terminate = bool(value)

    @property
    def only_execute_path_to_fetches(self) -> bool:
        """Set to True to execute only the nodes that the requested outputs depend on."""
        return self._native.get_config_entry(_ONLY_EXECUTE_PATH_TO_FETCHES) == "1"

    @only_execute_path_to_fetches.setter
    def only_execute_path_to_fetches(self, value: bool) -> None:
        self._native.add_config_entry(_ONLY_EXECUTE_PATH_TO_FETCHES, "1" if value else "0")

    def add_run_config_entry(self, config_key: str, config_value: str) -> None:
        """Set a run configuration entry; see onnxruntime_run_options_config_keys.h for the keys."""
        self._native.add_config_entry(config_key, config_value)

    def get_run_config_entry(self, config_key: str) -> str:
        """Return the value of a run configuration entry."""
        value = self._native.get_config_entry(config_key)
        if value is None:
            raise RuntimeError(f"RunOptions does not have configuration with key: {config_key}")
        return value

    def add_active_adapter(self, adapter: Any) -> None:
        """Activate the `LoraAdapter` `adapter` for the runs using these options."""
        self._native.add_active_lora_adapter(adapter._native)
        self._adapters.append(adapter)


# Key of `kOrtRunOptionsConfigOnlyExecutePathToFetches` in onnxruntime_run_options_config_keys.h.
_ONLY_EXECUTE_PATH_TO_FETCHES = "run.only_execute_path_to_fetches"


class OrtArenaCfg:
    """Configuration of an arena allocator; see `create_and_register_allocator`.

    Construct it either from `max_mem`, `arena_extend_strategy`, `initial_chunk_size_bytes`, and
    `max_dead_bytes_per_chunk`, or from a dict that may also hold `initial_growth_chunk_size_bytes` and
    `max_power_of_two_extend_bytes`. Unset values, given as -1, use the defaults of ONNX Runtime.
    """

    _KEYS = (
        "max_mem",
        "arena_extend_strategy",
        "initial_chunk_size_bytes",
        "max_dead_bytes_per_chunk",
        "initial_growth_chunk_size_bytes",
        "max_power_of_two_extend_bytes",
    )

    def __init__(self, *args: Any) -> None:
        self.max_mem = 0
        self.arena_extend_strategy = -1
        self.initial_chunk_size_bytes = -1
        self.max_dead_bytes_per_chunk = -1
        self.initial_growth_chunk_size_bytes = -1
        self.max_power_of_two_extend_bytes = -1
        if len(args) == 1 and isinstance(args[0], Mapping):
            for key, value in args[0].items():
                if key not in self._KEYS:
                    raise RuntimeError(f"Invalid OrtArenaCfg option: {key}")
                setattr(self, key, value)
        elif len(args) == 4:
            for key, value in zip(self._KEYS, args, strict=False):
                setattr(self, key, value)
        else:
            raise TypeError("OrtArenaCfg takes either 4 values or a dict")

    def _config(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in self._KEYS if getattr(self, key) >= 0}
