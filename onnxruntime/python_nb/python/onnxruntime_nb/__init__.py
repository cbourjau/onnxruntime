# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Prototype Python bindings for ONNX Runtime, built with nanobind on top of the public C API."""

# `has_collective_ops` of the pybind11 bindings is not provided: it depends on USE_MPI, which the build system never
# defines, so it is always false.
from ._C import get_all_providers, get_available_providers, get_device, set_seed
from ._environment import set_default_logger_severity, set_default_logger_verbosity
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
from ._session import ExecutionMode, GraphOptimizationLevel, InferenceSession, SessionOptions

__all__ = [
    "DeviceReset",
    "EPFail",
    "EngineError",
    "ExecutionMode",
    "Fail",
    "GraphOptimizationLevel",
    "InferenceSession",
    "InvalidArgument",
    "InvalidGraph",
    "InvalidProtobuf",
    "ModelLoadCanceled",
    "ModelLoaded",
    "ModelRequiresCompilation",
    "NoModel",
    "NoSuchFile",
    "NotFound",
    "NotImplemented",
    "RuntimeException",
    "SessionOptions",
    "get_all_providers",
    "get_available_providers",
    "get_device",
    "set_default_logger_severity",
    "set_default_logger_verbosity",
    "set_seed",
]
