# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Enumerations and the mapping between ONNX element types, numpy dtypes, and type strings."""

from __future__ import annotations

import enum
from typing import Any

import numpy as np


class GraphOptimizationLevel(enum.IntEnum):
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_LAYOUT = 3
    ORT_ENABLE_ALL = 99


class ExecutionMode(enum.IntEnum):
    ORT_SEQUENTIAL = 0
    ORT_PARALLEL = 1


class ExecutionOrder(enum.IntEnum):
    DEFAULT = 0
    PRIORITY_BASED = 1
    MEMORY_EFFICIENT = 2


class OrtAllocatorType(enum.IntEnum):
    INVALID = -1
    ORT_DEVICE_ALLOCATOR = 0
    ORT_ARENA_ALLOCATOR = 1


class OrtMemType(enum.IntEnum):
    CPU_INPUT = -2
    CPU_OUTPUT = -1
    CPU = -1
    DEFAULT = 0


class OrtMemoryInfoDeviceType(enum.IntEnum):
    CPU = 0
    GPU = 1
    FPGA = 2
    NPU = 3


class OrtDeviceMemoryType(enum.IntEnum):
    DEFAULT = 0
    HOST_ACCESSIBLE = 5


class OrtExecutionProviderDevicePolicy(enum.IntEnum):
    DEFAULT = 0
    PREFER_CPU = 1
    PREFER_NPU = 2
    PREFER_GPU = 3
    MAX_PERFORMANCE = 4
    MAX_EFFICIENCY = 5
    MIN_OVERALL_POWER = 6


class OrtHardwareDeviceType(enum.IntEnum):
    CPU = 0
    GPU = 1
    NPU = 2


class OrtSparseFormat(enum.IntEnum):
    ORT_SPARSE_UNDEFINED = 0
    ORT_SPARSE_COO = 1
    ORT_SPARSE_CSRC = 2
    ORT_SPARSE_BLOCK_SPARSE = 4


class OrtCompileApiFlags(enum.IntFlag):
    NONE = 0
    ERROR_IF_NO_NODES_COMPILED = 1
    ERROR_IF_OUTPUT_FILE_EXISTS = 2


class OrtCompiledModelCompatibility(enum.IntEnum):
    EP_NOT_APPLICABLE = 0
    EP_SUPPORTED_OPTIMAL = 1
    EP_SUPPORTED_PREFER_RECOMPILATION = 2
    EP_UNSUPPORTED = 3


class OrtDeviceEpIncompatibilityReason(enum.IntFlag):
    NONE = 0
    DRIVER_INCOMPATIBLE = 1
    DEVICE_INCOMPATIBLE = 2
    MISSING_DEPENDENCY = 4
    UNKNOWN = 1 << 31


class ArenaExtendStrategy(enum.IntEnum):
    kNextPowerOfTwo = 0  # noqa: N815
    kSameAsRequested = 1  # noqa: N815


class OrtDeviceVendorId(enum.IntEnum):
    """PCI vendor IDs of devices."""

    NONE = 0x0000
    AMD = 0x1002
    NVIDIA = 0x10DE
    ARM = 0x13B5
    MICROSOFT = 0x1414
    HUAWEI = 0x19E5
    QUALCOMM = 0x5143
    INTEL = 0x8086


# Values of `ONNXType` in onnxruntime_c_api.h.
ONNX_TYPE_TENSOR = 1
ONNX_TYPE_SEQUENCE = 2
ONNX_TYPE_MAP = 3
ONNX_TYPE_SPARSETENSOR = 5
ONNX_TYPE_OPTIONAL = 6

# Value of `ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING`, which equals `TensorProto.STRING`.
STRING = 8

# Names of the ONNX element types, indexed by their `TensorProto.DataType` value.
_NAMES = [
    "undefined",
    "float",
    "uint8",
    "int8",
    "uint16",
    "int16",
    "int32",
    "int64",
    "string",
    "bool",
    "float16",
    "double",
    "uint32",
    "uint64",
    "complex64",
    "complex128",
    "bfloat16",
    "float8e4m3fn",
    "float8e4m3fnuz",
    "float8e5m2",
    "float8e5m2fnuz",
    "uint4",
    "int4",
    "float4e2m1",
    "float8e8m0",
    "uint2",
    "int2",
    "float6e2m3",
    "float6e3m2",
]

# `ONNXTensorElementDataType` and `TensorProto.DataType` differ from 24 on.
_NATIVE_TO_ONNX = {24: 25, 25: 26, 26: 24}
_ONNX_TO_NATIVE = {v: k for k, v in _NATIVE_TO_ONNX.items()}

_NUMPY_TYPES = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    8: np.object_,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
}
_FROM_NUMPY = {np.dtype(t): onnx_type for onnx_type, t in _NUMPY_TYPES.items() if t is not np.object_}


def to_onnx(native_type: int) -> int:
    """Return the `TensorProto.DataType` of an `ONNXTensorElementDataType`."""
    return _NATIVE_TO_ONNX.get(native_type, native_type)


def to_native(onnx_type: int) -> int:
    """Return the `ONNXTensorElementDataType` of a `TensorProto.DataType`."""
    return _ONNX_TO_NATIVE.get(onnx_type, onnx_type)


def numpy_dtype(native_type: int) -> np.dtype:
    """Return the numpy dtype of an `ONNXTensorElementDataType`."""
    try:
        return np.dtype(_NUMPY_TYPES[native_type])
    except KeyError:
        raise TypeError(f"element type `{type_string(('tensor', native_type, None))}` has no numpy dtype") from None


def element_type(dtype: Any) -> int:
    """Return the `ONNXTensorElementDataType` of a numpy dtype or a `TensorProto.DataType` value."""
    if isinstance(dtype, int) and not isinstance(dtype, np.generic):
        return to_native(dtype)
    dtype = np.dtype(dtype)
    if dtype.kind in "OUS":
        return STRING
    try:
        return _FROM_NUMPY[dtype]
    except KeyError:
        raise TypeError(f"numpy dtype `{dtype}` is not supported") from None


def type_string(description: tuple) -> str:
    """Return the ONNX type string, e.g. 'seq(tensor(float))', of a type description of the native module."""
    kind = description[0]
    if kind in ("tensor", "sparse_tensor"):
        return f"{kind}({_NAMES[to_onnx(description[1])]})"
    if kind == "seq":
        return f"seq({type_string(description[1])})"
    if kind == "map":
        return f"map({_NAMES[to_onnx(description[1])]},{type_string(description[2])})"
    if kind == "optional":
        return f"optional({type_string(description[1])})"
    return kind
