# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Python bindings for ONNX Runtime, built with nanobind on top of the public C API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .capi import _C
from ._adapter import AdapterFormat, LoraAdapter
from .capi._C import (
    OrtEpContextData,
    OrtEpContextDataBuffer,
    OrtEpDevice,
    OrtHardwareDevice,
    OrtMemoryInfo,
    OrtSyncStream,
    disable_telemetry_events,
    enable_telemetry_events,
    get_all_providers,
    get_available_providers,
    get_build_info,
    get_compatibility_info_from_model,
    get_compatibility_info_from_model_bytes,
    get_device,
    get_ep_devices,
    get_hardware_device_ep_incompatibility_details,
    get_hardware_devices,
    get_version_string,
    register_execution_provider_library,
    set_default_logger_verbosity,
    set_seed,
    unregister_execution_provider_library,
)
from ._compile import ModelCompiler, OrtExternalInitializerInfo
from ._exceptions import (
    DeviceReset,
    EngineError,
    EPFail,
    Fail,
    InvalidArgument,
    InvalidGraph,
    InvalidProtobuf,
    ModelLoadCanceled,
    ModelLoaded,
    ModelRequiresCompilation,
    NoModel,
    NoSuchFile,
    NotFound,
    NotImplemented,
    RuntimeException,
)
from ._model_package import (
    ModelPackage,
    ModelPackageComponentContext,
    ModelPackageContext,
    ModelPackageOptions,
    ModelPackageWriteMode,
)
from ._options import OrtArenaCfg, RunOptions, SessionOptions
from ._session import (
    InferenceSession,
    IOBinding,
    ModelMetadata,
    NodeArg,
    OrtEpAssignedNode,
    OrtEpAssignedSubgraph,
    copy_tensors,
)
from ._types import (
    ArenaExtendStrategy,
    ExecutionMode,
    ExecutionOrder,
    GraphOptimizationLevel,
    OrtAllocatorType,
    OrtCompileApiFlags,
    OrtCompiledModelCompatibility,
    OrtDeviceEpIncompatibilityReason,
    OrtDeviceMemoryType,
    OrtDeviceVendorId,
    OrtExecutionProviderDevicePolicy,
    OrtHardwareDeviceType,
    OrtMemoryInfoDeviceType,
    OrtMemType,
    OrtSparseFormat,
)
from ._value import OrtDevice, OrtValue, SparseTensor

__version__ = get_version_string()

# Same as SessionIOBinding of the pybind11 bindings.
SessionIOBinding = IOBinding


def has_collective_ops() -> bool:
    """Return whether this build has the collective operators; they require MPI, which these bindings lack."""
    return False


def set_default_logger_severity(severity: int) -> None:
    """Set the severity of the default logger: 0 (verbose), 1 (info), 2 (warning), 3 (error), or 4 (fatal)."""
    if not 0 <= severity <= 4:
        raise RuntimeError("Invalid logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal")
    _C.set_default_logger_severity(severity)


def set_global_thread_pool_sizes(intra_op_num_threads: int = 0, inter_op_num_threads: int = 0) -> None:
    """Make sessions share thread pools of the given sizes; 0 lets ONNX Runtime choose.

    Call it before creating any session. Sessions use the shared thread pools if their options have
    `use_per_session_threads` set to False.
    """
    _C.set_global_thread_pool_sizes(intra_op_num_threads, inter_op_num_threads)


def create_and_register_allocator(mem_info: OrtMemoryInfo, arena_cfg: OrtArenaCfg | None = None) -> None:
    """Create an allocator for `mem_info` that sessions share if they set 'session.use_env_allocators' to '1'."""
    _C.create_and_register_allocator(mem_info, None if arena_cfg is None else arena_cfg._config())


def create_and_register_allocator_v2(
    provider_type: str,
    mem_info: OrtMemoryInfo,
    provider_options: Mapping[str, Any],
    arena_cfg: OrtArenaCfg | None = None,
) -> None:
    """Like `create_and_register_allocator`, with the allocator of the execution provider `provider_type`."""
    options = {str(k): str(v) for k, v in provider_options.items()}
    _C.create_and_register_allocator_v2(
        provider_type, mem_info, options, None if arena_cfg is None else arena_cfg._config()
    )


def get_model_compatibility_for_ep_devices(
    ep_devices: list[OrtEpDevice], compatibility_info: str
) -> OrtCompiledModelCompatibility:
    """Return the compatibility of a compiled model, given its compatibility info, with `ep_devices`."""
    return OrtCompiledModelCompatibility(_C.get_model_compatibility_for_ep_devices(ep_devices, compatibility_info))


__all__ = [
    "AdapterFormat",
    "ArenaExtendStrategy",
    "DeviceReset",
    "EPFail",
    "EngineError",
    "ExecutionMode",
    "ExecutionOrder",
    "Fail",
    "GraphOptimizationLevel",
    "IOBinding",
    "InferenceSession",
    "InvalidArgument",
    "InvalidGraph",
    "InvalidProtobuf",
    "LoraAdapter",
    "ModelCompiler",
    "ModelLoadCanceled",
    "ModelLoaded",
    "ModelMetadata",
    "ModelPackage",
    "ModelPackageComponentContext",
    "ModelPackageContext",
    "ModelPackageOptions",
    "ModelPackageWriteMode",
    "ModelRequiresCompilation",
    "NoModel",
    "NoSuchFile",
    "NodeArg",
    "NotFound",
    "NotImplemented",
    "OrtAllocatorType",
    "OrtArenaCfg",
    "OrtCompileApiFlags",
    "OrtCompiledModelCompatibility",
    "OrtDevice",
    "OrtDeviceEpIncompatibilityReason",
    "OrtDeviceMemoryType",
    "OrtDeviceVendorId",
    "OrtEpAssignedNode",
    "OrtEpAssignedSubgraph",
    "OrtEpContextData",
    "OrtEpContextDataBuffer",
    "OrtEpDevice",
    "OrtExecutionProviderDevicePolicy",
    "OrtExternalInitializerInfo",
    "OrtHardwareDevice",
    "OrtHardwareDeviceType",
    "OrtMemType",
    "OrtMemoryInfo",
    "OrtMemoryInfoDeviceType",
    "OrtSparseFormat",
    "OrtSyncStream",
    "OrtValue",
    "RunOptions",
    "RuntimeException",
    "SessionIOBinding",
    "SessionOptions",
    "SparseTensor",
    "copy_tensors",
    "create_and_register_allocator",
    "create_and_register_allocator_v2",
    "disable_telemetry_events",
    "enable_telemetry_events",
    "get_all_providers",
    "get_available_providers",
    "get_build_info",
    "get_compatibility_info_from_model",
    "get_compatibility_info_from_model_bytes",
    "get_device",
    "get_ep_devices",
    "get_hardware_device_ep_incompatibility_details",
    "get_hardware_devices",
    "get_model_compatibility_for_ep_devices",
    "get_version_string",
    "has_collective_ops",
    "register_execution_provider_library",
    "set_default_logger_severity",
    "set_default_logger_verbosity",
    "set_global_thread_pool_sizes",
    "set_seed",
    "unregister_execution_provider_library",
]
