# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Values that sessions consume and produce, and their conversion from and to Python objects."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from .capi import _C
from ._types import (
    ONNX_TYPE_MAP,
    ONNX_TYPE_SEQUENCE,
    ONNX_TYPE_SPARSETENSOR,
    ONNX_TYPE_TENSOR,
    STRING,
    OrtAllocatorType,
    OrtDeviceVendorId,
    OrtSparseFormat,
    element_type,
    numpy_dtype,
    to_native,
    to_onnx,
    type_string,
)


class OrtDevice:
    """A device and the kind of its memory that hold the data of a value.

    Parameters
    ----------
    device_type
        One of the device types `cpu()`, `gpu()`, `npu()`, `fpga()`, or `dml()`.
    mem_type
        0 for device memory or 5 for host-accessible memory.
    vendor_id
        PCI vendor ID of the device.
    device_id
        Index of the device among the devices of its type.
    """

    _CPU, _GPU, _FPGA, _NPU, _DML = 0, 1, 2, 3, 4
    # Device type and vendor of device names; the vendor of 'gpu' and 'npu' is given explicitly.
    _NAMES: ClassVar[dict[str, tuple[int, int]]] = {
        "cpu": (_CPU, OrtDeviceVendorId.NONE),
        "cuda": (_GPU, OrtDeviceVendorId.NVIDIA),
        "dml": (_GPU, OrtDeviceVendorId.MICROSOFT),
        "cann": (_NPU, OrtDeviceVendorId.HUAWEI),
        "webgpu": (_GPU, OrtDeviceVendorId.NONE),
        "gpu": (_GPU, OrtDeviceVendorId.NONE),
        "npu": (_NPU, OrtDeviceVendorId.NONE),
        "fpga": (_FPGA, OrtDeviceVendorId.NONE),
    }
    # Allocator names of `OrtMemoryInfo` by vendor; see `onnxruntime::CUDA` and friends in allocator.h.
    _ALLOCATOR_NAMES: ClassVar[dict[int, str]] = {
        OrtDeviceVendorId.NVIDIA: "Cuda",
        OrtDeviceVendorId.MICROSOFT: "DML",
        OrtDeviceVendorId.HUAWEI: "Cann",
    }

    def __init__(self, device_type: int, mem_type: int = 0, vendor_id: int = 0, device_id: int = 0):
        self._device_type = int(device_type)
        self._mem_type = int(mem_type)
        self._vendor_id = int(vendor_id)
        self._device_id = int(device_id)

    @staticmethod
    def make(ort_device_name: str, device_id: int, vendor_id: int = -1) -> OrtDevice:
        """Return the device with name 'cpu', 'cuda', 'dml', 'cann', 'webgpu', 'gpu', or 'npu'."""
        try:
            device_type, default_vendor = OrtDevice._NAMES[ort_device_name.lower()]
        except KeyError:
            raise Exception(f"Unsupported device type: {ort_device_name}") from None
        return OrtDevice(device_type, 0, default_vendor if vendor_id < 0 else vendor_id, device_id)

    @staticmethod
    def cpu() -> int:
        return OrtDevice._CPU

    @staticmethod
    def gpu() -> int:
        return OrtDevice._GPU

    @staticmethod
    def npu() -> int:
        return OrtDevice._NPU

    @staticmethod
    def fpga() -> int:
        return OrtDevice._FPGA

    @staticmethod
    def cuda() -> int:
        return OrtDevice._GPU

    @staticmethod
    def dml() -> int:
        return OrtDevice._DML

    @staticmethod
    def default_memory() -> int:
        return 0

    def device_id(self) -> int:
        return self._device_id

    def device_type(self) -> int:
        return self._device_type

    def device_vendor_id(self) -> int:
        return self._vendor_id

    def device_mem_type(self) -> int:
        return self._mem_type

    def _memory_info(self) -> _C.OrtMemoryInfo:
        device_type, vendor = self._device_type, self._vendor_id
        if device_type == self._DML:
            device_type, vendor = self._GPU, OrtDeviceVendorId.MICROSOFT
        if device_type == self._CPU:
            name = "Cpu"
        else:
            name = self._ALLOCATOR_NAMES.get(vendor, "Gpu" if device_type == self._GPU else "Npu")
        return _C.OrtMemoryInfo.create_v2(
            name, device_type, vendor, self._device_id, self._mem_type, 0, OrtAllocatorType.ORT_DEVICE_ALLOCATOR
        )


class OrtValue:
    """A value of any ONNX type, e.g. a tensor, whose data may reside on a device such as a GPU.

    Values created from numpy arrays on the CPU borrow the buffer of the array.
    """

    def __init__(self, value: _C.OrtValue, session: Any = None):
        if not isinstance(value, _C.OrtValue):
            raise TypeError("use the factory methods of 'OrtValue' to create values")
        self._value = value
        # Session whose allocator owns the buffer, if any.
        self._session = session

    @classmethod
    def ortvalue_from_numpy(
        cls, numpy_obj: np.ndarray, /, device_type: str = "cpu", device_id: int = 0, vendor_id: int = -1
    ) -> OrtValue:
        """Return a tensor with the data of `numpy_obj` on the given device.

        On the CPU, the tensor borrows the buffer of C-contiguous numeric arrays; otherwise, it holds a copy.
        """
        value = tensor_from_array(numpy_obj)
        if device_type.lower() != "cpu":
            device_value = _C.OrtValue.empty(
                value.shape, value.element_type, OrtDevice.make(device_type, device_id, vendor_id)._memory_info()
            )
            device_value.copy_from(value)
            value = device_value
        return cls(value)

    @classmethod
    def ortvalue_from_numpy_with_onnx_type(cls, data: np.ndarray, /, onnx_element_type: int) -> OrtValue:
        """Return a tensor of `TensorProto.DataType` `onnx_element_type` that borrows the buffer of `data`.

        Use it for element types that numpy lacks, such as bfloat16; the buffer is reinterpreted.
        """
        return cls(_C.OrtValue.from_array(np.ascontiguousarray(data), to_native(onnx_element_type)))

    @classmethod
    def ortvalue_from_shape_and_type(
        cls,
        shape: Sequence[int],
        element_type: Any,
        device_type: str = "cpu",
        device_id: int = 0,
        vendor_id: int = -1,
        memory_info: _C.OrtMemoryInfo | None = None,
    ) -> OrtValue:
        """Return an uninitialized tensor allocated by the shared allocator of a device.

        `element_type` is a numpy dtype or a `TensorProto.DataType` value. If given, `memory_info` selects the
        allocator instead of the device.
        """
        if memory_info is None:
            memory_info = OrtDevice.make(device_type, device_id, vendor_id)._memory_info()
        elif device_type != "cpu" or device_id != 0 or vendor_id != -1:
            warnings.warn(
                "device_type, device_id, and vendor_id are ignored when memory_info is provided.", stacklevel=2
            )
        return cls(_C.OrtValue.empty(list(shape), _element_type(element_type), memory_info))

    @classmethod
    def ort_value_from_sparse_tensor(cls, sparse_tensor: SparseTensor) -> OrtValue:
        """Return a value holding `sparse_tensor`."""
        return cls(sparse_tensor._value)

    @classmethod
    def from_dlpack(cls, data: Any, /) -> OrtValue:
        """Return a tensor that borrows the buffer of a DLPack capsule or an object with `__dlpack__`."""
        if hasattr(data, "__dlpack__"):
            data = data.__dlpack__()
        return cls(_C.OrtValue.from_dlpack(data))

    def as_sparse_tensor(self) -> SparseTensor:
        """Return the sparse tensor held by this value."""
        return SparseTensor(self._value)

    def data_ptr(self) -> int:
        """Return the address of the buffer of the tensor."""
        return self._value.data_ptr

    def device_name(self) -> str:
        """Return the lower-case name of the device of the buffer, e.g. 'cpu' or 'cuda'."""
        return self._value.memory_info.name.lower()

    def shape(self) -> list[int]:
        return self._value.shape

    def data_type(self) -> str:
        """Return the ONNX type string, e.g. 'tensor(int64)'."""
        return type_string(self._value.type)

    def element_type(self) -> int:
        """Return the `TensorProto.DataType` of the elements of the tensor."""
        return to_onnx(self._value.element_type)

    def tensor_size_in_bytes(self) -> int:
        return self._value.size_in_bytes

    def has_value(self) -> bool:
        """Return False if this is an empty optional value."""
        return self._value.has_value

    def is_tensor(self) -> bool:
        return self._value.onnx_type == ONNX_TYPE_TENSOR

    def is_sparse_tensor(self) -> bool:
        return self._value.onnx_type == ONNX_TYPE_SPARSETENSOR

    def is_tensor_sequence(self) -> bool:
        return self._value.onnx_type == ONNX_TYPE_SEQUENCE

    def numpy(self) -> np.ndarray:
        """Return the tensor as a numpy array; numeric tensors on the CPU are viewed, not copied."""
        if not self.is_tensor():
            raise ValueError(f"only tensors can be converted to numpy arrays, not `{self.data_type()}`")
        return _tensor_to_numpy(self._value)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        array = self.numpy()
        if copy is not None:
            return np.array(array, dtype=dtype, copy=copy)
        return array if dtype is None else np.asarray(array, dtype=dtype)

    def __dlpack__(self, *, stream: Any = None, **kwargs: Any) -> Any:
        return self._value.__dlpack__()

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._value.__dlpack_device__()

    def update_inplace(self, data: np.ndarray | OrtValue) -> None:
        """Copy `data`, which has the shape and type of this tensor, into the buffer of this tensor."""
        if isinstance(data, OrtValue):
            if data._session is not None and self._session is not None and data._session is not self._session:
                raise ValueError("Session-scoped OrtValues must originate from the same session.")
            self._value.copy_from(data._value)
            return
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray or an OrtValue.")
        source = tensor_from_array(data)
        if source.shape != self._value.shape or source.element_type != self._value.element_type:
            raise ValueError(
                f"the data with shape {source.shape} and element type {source.element_type} does not match the "
                f"tensor with shape {self._value.shape} and element type {self._value.element_type}"
            )
        self._value.copy_from(source)


class SparseTensor:
    """A sparse tensor in COO, CSR, or block-sparse format.

    Sparse tensors created from numeric numpy arrays borrow their buffers.
    """

    def __init__(self, value: _C.OrtValue):
        if not isinstance(value, _C.OrtValue):
            raise TypeError("use the factory methods of 'SparseTensor' to create sparse tensors")
        self._value = value

    @classmethod
    def sparse_coo_from_numpy(
        cls, dense_shape: Any, values: np.ndarray, coo_indices: np.ndarray, ort_device: OrtDevice
    ) -> SparseTensor:
        """Return a sparse tensor in COO format.

        `coo_indices` holds either the linear index of each value or the coordinate pairs of each value of a 2-D
        tensor.
        """
        _check_cpu(ort_device)
        return cls(_C.OrtValue.sparse_coo(_dense_shape(dense_shape), tensor_from_array(values), _indices(coo_indices)))

    @classmethod
    def sparse_csr_from_numpy(
        cls,
        dense_shape: Any,
        values: np.ndarray,
        inner_indices: np.ndarray,
        outer_indices: np.ndarray,
        ort_device: OrtDevice,
    ) -> SparseTensor:
        """Return a 2-D sparse tensor in CSR format."""
        _check_cpu(ort_device)
        return cls(
            _C.OrtValue.sparse_csr(
                _dense_shape(dense_shape),
                tensor_from_array(values),
                _indices(inner_indices),
                _indices(outer_indices),
            )
        )

    def values(self) -> np.ndarray:
        """Return the non-zero values; numeric values are viewed, not copied."""
        values = self._value.sparse_values()
        return _object_array(values, [len(values)]) if isinstance(values, list) else _read_only(values)

    def as_coo_view(self) -> _IndicesView:
        """Return a view of the COO indices; raises if the format is not COO."""
        self._check_format(OrtSparseFormat.ORT_SPARSE_COO)
        return _IndicesView(indices=_read_only(self._value.sparse_indices(0)))

    def as_csrc_view(self) -> _IndicesView:
        """Return a view of the CSR inner and outer indices; raises if the format is not CSR."""
        self._check_format(OrtSparseFormat.ORT_SPARSE_CSRC)
        return _IndicesView(
            inner=_read_only(self._value.sparse_indices(1)), outer=_read_only(self._value.sparse_indices(2))
        )

    def as_blocksparse_view(self) -> _IndicesView:
        """Return a view of the block-sparse indices; raises if the format is not block-sparse."""
        self._check_format(OrtSparseFormat.ORT_SPARSE_BLOCK_SPARSE)
        return _IndicesView(indices=_read_only(self._value.sparse_indices(3)))

    def to_cuda(self, ort_device: OrtDevice) -> SparseTensor:
        raise NotImplementedError("copying sparse tensors to CUDA devices is not supported")

    def format(self) -> OrtSparseFormat:
        return OrtSparseFormat(self._value.sparse_format)

    def dense_shape(self) -> list[int]:
        return self._value.shape

    def data_type(self) -> str:
        """Return the ONNX type string, e.g. 'sparse_tensor(float)'."""
        return type_string(("sparse_tensor", self._value.sparse_values_type[0]))

    def device_name(self) -> str:
        return "cpu"

    def _check_format(self, expected: OrtSparseFormat) -> None:
        if self.format() != expected:
            raise RuntimeError(f"the sparse tensor has format `{self.format().name}`, not `{expected.name}`")


class _IndicesView:
    """Indices of a sparse tensor; call the attributes to get them."""

    def __init__(self, **indices: np.ndarray):
        for name, array in indices.items():
            setattr(self, name, lambda array=array: array)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _check_cpu(ort_device: OrtDevice) -> None:
    if ort_device.device_type() != OrtDevice.cpu():
        raise NotImplementedError("sparse tensors are only supported on the CPU")


def _dense_shape(dense_shape: Any) -> list[int]:
    return [int(d) for d in np.asarray(dense_shape).ravel()]


def _indices(indices: np.ndarray) -> _C.OrtValue:
    if not isinstance(indices, np.ndarray) or indices.dtype != np.int64:
        raise TypeError("sparse tensor indices must be a numpy array of dtype int64")
    return tensor_from_array(indices)


def _element_type(dtype: Any) -> int:
    """Return the `ONNXTensorElementDataType` of a numpy dtype or a `TensorProto.DataType` value."""
    return element_type(dtype)


def _object_array(elements: list, shape: list[int]) -> np.ndarray:
    array = np.empty(len(elements), dtype=object)
    array[:] = elements
    return array.reshape(shape)


def tensor_from_array(array: Any) -> _C.OrtValue:
    """Return a tensor that borrows the buffer of numeric arrays and copies string arrays."""
    array = np.asarray(array)
    if array.dtype.kind in "OUSV":
        strings = [_to_bytes(x) for x in array.ravel().tolist()]
        return _C.OrtValue.from_strings(strings, list(array.shape))
    return _C.OrtValue.from_array(np.ascontiguousarray(array))


def _to_bytes(x: Any) -> bytes:
    return bytes(x) if isinstance(x, (bytes, bytearray)) else str(x).encode()


def _tensor_to_numpy(value: _C.OrtValue) -> np.ndarray:
    if value.element_type == STRING:
        return _object_array(value.strings(), value.shape)
    return value.numpy()


def to_python(value: _C.OrtValue) -> Any:
    """Return a numpy array, list, dict, `SparseTensor`, or None for an output value of a session."""
    if not value.has_value:
        return None
    onnx_type = value.onnx_type
    if onnx_type == ONNX_TYPE_TENSOR:
        return _tensor_to_numpy(value)
    if onnx_type == ONNX_TYPE_SEQUENCE:
        return [to_python(value.element(i)) for i in range(value.count)]
    if onnx_type == ONNX_TYPE_MAP:
        keys = _tensor_to_numpy(value.element(0)).tolist()
        values = _tensor_to_numpy(value.element(1)).tolist()
        return dict(zip(keys, values, strict=True))
    if onnx_type == ONNX_TYPE_SPARSETENSOR:
        return SparseTensor(value)
    raise NotImplementedError(f"outputs of type `{type_string(value.type)}` are not supported")


def from_python(obj: Any, description: tuple) -> _C.OrtValue:
    """Return a value of the type `description` of the native module with the data of `obj`."""
    if isinstance(obj, OrtValue):
        return obj._value
    if isinstance(obj, SparseTensor):
        return obj._value
    kind = description[0]
    if kind == "optional":
        return from_python(obj, description[1])
    if kind == "seq" and isinstance(obj, (list, tuple)):
        if not obj:
            return _empty_sequence(description[1])
        return _C.OrtValue.sequence([from_python(element, description[1]) for element in obj])
    if kind == "map" and isinstance(obj, dict):
        return _map(obj, description)
    if kind == "seq" and isinstance(obj, dict):
        raise TypeError("a sequence input requires a list")
    if isinstance(obj, np.ndarray):
        return tensor_from_array(obj)
    if kind == "tensor":
        try:
            dtype = numpy_dtype(description[1])
        except TypeError:
            dtype = None
        return tensor_from_array(np.asarray(obj, dtype=dtype if dtype != np.object_ else None))
    raise TypeError(f"cannot convert an object of type '{type(obj).__name__}' to `{type_string(description)}`")


def _map(obj: dict, description: tuple) -> _C.OrtValue:
    if not obj:
        raise ValueError("maps must not be empty")
    _, key_type, value_description = description
    key_dtype = numpy_dtype(key_type)
    for key in obj:
        if key_dtype != np.object_ and not isinstance(key, (int, np.integer)):
            raise RuntimeError(f"Unexpected key type  {type(key)}, it cannot be linked to C type int64_t")
    keys = np.array(list(obj.keys()), dtype=key_dtype)
    value_type = value_description[1] if value_description[0] == "tensor" else None
    values = np.array(list(obj.values()), dtype=numpy_dtype(value_type) if value_type else None)
    return _C.OrtValue.map(tensor_from_array(keys), tensor_from_array(values))


_EMPTY_SEQUENCE_SESSIONS: dict[int, _C.Session] = {}


def _empty_sequence(element_description: tuple) -> _C.OrtValue:
    """Return an empty sequence of tensors of the element type of `element_description`.

    The C API cannot create empty sequences, so a session with a single 'SequenceEmpty' node creates them.
    """
    if element_description[0] != "tensor":
        raise NotImplementedError(f"empty sequences of `{type_string(element_description)}` are not supported")
    native_type = element_description[1]
    session = _EMPTY_SEQUENCE_SESSIONS.get(native_type)
    if session is None:
        options = _C.SessionOptions()
        session = _C.Session(_sequence_empty_model(to_onnx(native_type)), options)
        _EMPTY_SEQUENCE_SESSIONS[native_type] = session
    return session.run([], [], ["s"], None)[0]


def _sequence_empty_model(onnx_type: int) -> bytes:
    """Return a serialized ONNX model whose output 's' is an empty sequence of tensors of `onnx_type`."""

    def varint(n: int) -> bytes:
        out = bytearray()
        while True:
            byte, n = n & 0x7F, n >> 7
            out.append(byte | (0x80 if n else 0))
            if not n:
                return bytes(out)

    def field(number: int, payload: bytes | int) -> bytes:
        if isinstance(payload, int):
            return varint(number << 3) + varint(payload)
        return varint(number << 3 | 2) + varint(len(payload)) + payload

    # Field numbers of onnx.proto: AttributeProto, NodeProto, TypeProto, ValueInfoProto, GraphProto, ModelProto.
    attribute = field(1, b"dtype") + field(3, onnx_type) + field(20, 2)
    node = field(2, b"s") + field(4, b"SequenceEmpty") + field(5, attribute)
    sequence_type = field(4, field(1, field(1, field(1, onnx_type))))
    graph = field(1, node) + field(2, b"g") + field(12, field(1, b"s") + field(2, sequence_type))
    return field(1, 8) + field(7, graph) + field(8, field(1, b"") + field(2, 11))
