// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "value.h"

#include <cstring>
#include <limits>
#include <optional>
#include <utility>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace onnxruntime_nb {

namespace {

using nb::dlpack::dtype_code;

constexpr nb::dlpack::dtype Dtype(dtype_code code, int bits) {
  return {static_cast<uint8_t>(code), static_cast<uint8_t>(bits), 1};
}

const std::pair<ONNXTensorElementDataType, nb::dlpack::dtype> kElementTypes[] = {
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, Dtype(dtype_code::Float, 32)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, Dtype(dtype_code::Float, 64)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, Dtype(dtype_code::Float, 16)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, Dtype(dtype_code::Bfloat, 16)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, Dtype(dtype_code::Int, 8)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, Dtype(dtype_code::Int, 16)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, Dtype(dtype_code::Int, 32)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, Dtype(dtype_code::Int, 64)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, Dtype(dtype_code::UInt, 8)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, Dtype(dtype_code::UInt, 16)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, Dtype(dtype_code::UInt, 32)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, Dtype(dtype_code::UInt, 64)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, Dtype(dtype_code::Bool, 8)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN, Dtype(dtype_code::Float8_E4M3FN, 8)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ, Dtype(dtype_code::Float8_E4M3FNUZ, 8)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2, Dtype(dtype_code::Float8_E5M2, 8)},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ, Dtype(dtype_code::Float8_E5M2FNUZ, 8)},
};

// DLPack device types; see `DLDeviceType` in dlpack.h.
constexpr int kDLCPU = 1;
constexpr int kDLCUDA = 2;

// Arrays of any device and layout; the layout is checked where it matters.
using AnyArray = nb::ndarray<nb::ro>;

ONNXTensorElementDataType ElementType(nb::dlpack::dtype dtype) {
  for (const auto& [onnx_type, dl_type] : kElementTypes) {
    if (dl_type == dtype) {
      return onnx_type;
    }
  }
  throw Ort::Exception("arrays with DLPack dtype code " + std::to_string(dtype.code) + " and " +
                           std::to_string(dtype.bits) + " bits are not supported",
                       ORT_INVALID_ARGUMENT);
}

nb::dlpack::dtype DlpackType(ONNXTensorElementDataType element_type) {
  for (const auto& [onnx_type, dl_type] : kElementTypes) {
    if (onnx_type == element_type) {
      return dl_type;
    }
  }
  throw Ort::Exception("tensors of element type " + std::to_string(element_type) + " cannot be viewed as arrays",
                       ORT_NOT_IMPLEMENTED);
}

Ort::MemoryInfo MemoryInfoForDlpackDevice(int device_type, int device_id) {
  switch (device_type) {
    case kDLCPU:
      return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    case kDLCUDA:
      return Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device_id, OrtMemTypeDefault);
    default:
      throw Ort::Exception("arrays on DLPack device type " + std::to_string(device_type) + " are not supported",
                           ORT_NOT_IMPLEMENTED);
  }
}

std::pair<int, int> DlpackDevice(Ort::ConstMemoryInfo memory_info) {
  if (memory_info.GetDeviceType() == OrtMemoryInfoDeviceType_CPU ||
      memory_info.GetDeviceMemoryType() == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
    return {kDLCPU, 0};
  }
  if (memory_info.GetDeviceType() == OrtMemoryInfoDeviceType_GPU &&
      memory_info.GetVendorId() == 0x10DE /* NVIDIA */) {
    return {kDLCUDA, memory_info.GetDeviceId()};
  }
  throw Ort::Exception("tensors on device '" + memory_info.GetAllocatorName() + "' cannot be exported via DLPack",
                       ORT_NOT_IMPLEMENTED);
}

Value& Unwrap(nb::handle self) { return nb::cast<Value&>(self); }

Value TensorFromArray(const AnyArray& array, nb::handle owner, std::optional<int> element_type) {
  int64_t expected_stride = 1;
  for (size_t i = array.ndim(); i-- > 0;) {
    if (array.shape(i) != 1 && array.stride(i) != expected_stride && array.size() != 0) {
      throw Ort::Exception("arrays must be C-contiguous", ORT_INVALID_ARGUMENT);
    }
    expected_stride *= array.shape(i);
  }
  auto memory_info = MemoryInfoForDlpackDevice(array.device_type(), array.device_id());
  std::vector<int64_t> shape(array.shape_ptr(), array.shape_ptr() + array.ndim());
  auto type = element_type ? static_cast<ONNXTensorElementDataType>(*element_type) : ElementType(array.dtype());
  auto value = Ort::Value::CreateTensor(memory_info, const_cast<void*>(array.data()), array.nbytes(), shape.data(),
                                        shape.size(), type);
  return {nb::borrow(owner), std::move(value)};
}

// Returns a string tensor with copies of the UTF-8 encoded `strings`, which may contain null characters.
Value TensorFromStrings(const std::vector<nb::bytes>& strings, const std::vector<int64_t>& shape) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto value = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
  for (size_t i = 0; i < strings.size(); ++i) {
    char* buffer = value.GetResizedStringTensorElementBuffer(i, strings[i].size());
    std::memcpy(buffer, strings[i].c_str(), strings[i].size());
  }
  return {{}, std::move(value)};
}

nb::object TensorToArray(nb::handle self, bool numpy) {
  Ort::Value& value = Unwrap(self).value;
  auto info = value.GetTensorTypeAndShapeInfo();
  auto dtype = DlpackType(info.GetElementType());
  auto [device_type, device_id] = DlpackDevice(value.GetTensorMemoryInfo());
  std::vector<int64_t> dims = info.GetShape();
  std::vector<size_t> shape(dims.begin(), dims.end());
  void* data = value.GetTensorMutableRawData();
  if (numpy) {
    if (device_type != kDLCPU) {
      throw Ort::Exception("only tensors on the CPU can be viewed as numpy arrays", ORT_INVALID_ARGUMENT);
    }
    return nb::ndarray<nb::numpy>(data, shape.size(), shape.data(), self, nullptr, dtype, device_type).cast();
  }
  return nb::ndarray<>(data, shape.size(), shape.data(), self, nullptr, dtype, device_type, device_id).cast();
}

std::vector<int64_t> SparseIndicesShape(const Ort::Value& value, OrtSparseIndicesFormat format) {
  return value.GetSparseTensorIndicesTypeShapeInfo(format).GetShape();
}

// Returns a sparse tensor with the dense shape `dense_shape` and the elements of the 1-D tensor `values`.
// Numeric values are borrowed; strings are copied by `fill`.
template <typename UseIndices, typename Fill>
Value SparseTensor(const std::vector<int64_t>& dense_shape, const Value& values, nb::object owner,
                   UseIndices use_indices, Fill fill) {
  auto info = values.value.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> values_shape = info.GetShape();
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    std::vector<std::string> strings = GetStrings(values.value, info.GetElementCount());
    std::vector<const char*> pointers;
    for (const auto& s : strings) {
      pointers.push_back(s.c_str());
    }
    Ort::AllocatorWithDefaultOptions allocator;
    auto value = Ort::Value::CreateSparseTensor(allocator, {dense_shape.data(), dense_shape.size()},
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    auto cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::detail::OrtSparseValuesParam param{values_shape.data(), values_shape.size(), {}};
    param.data.str = pointers.data();
    fill(value, cpu, param);
    return {{}, std::move(value)};
  }
  OrtValue* out = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateSparseTensorWithValuesAsOrtValue(
      values.value.GetTensorMemoryInfo(), const_cast<void*>(values.value.GetTensorRawData()), dense_shape.data(),
      dense_shape.size(), values_shape.data(), values_shape.size(), info.GetElementType(), &out));
  Ort::Value value(out);
  use_indices(value);
  return {owner, std::move(value)};
}

int64_t* MutableData(const Value& value) {
  return const_cast<int64_t*>(value.value.GetTensorData<int64_t>());
}

size_t ElementCount(const Value& value) { return value.value.GetTensorTypeAndShapeInfo().GetElementCount(); }

}  // namespace

OrtAllocator* GetAllocator(const OrtMemoryInfo* memory_info) {
  Ort::ConstMemoryInfo info{memory_info};
  if (info.GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
    Ort::AllocatorWithDefaultOptions allocator;
    return allocator;
  }
  OrtAllocator* allocator = nullptr;
  Ort::ThrowOnError(Ort::GetApi().GetSharedAllocator(GetEnv(), memory_info, &allocator));
  if (allocator == nullptr) {
    throw Ort::Exception("No shared allocator found for device '" + info.GetAllocatorName() + "'",
                         ORT_INVALID_ARGUMENT);
  }
  return allocator;
}

std::vector<std::string> GetStrings(const OrtValue* value, size_t count) {
  Ort::ConstValue v{value};
  size_t length = v.GetStringTensorDataLength();
  std::string buffer(length, '\0');
  std::vector<size_t> offsets(count);
  if (count > 0) {
    v.GetStringTensorContent(buffer.data(), length, offsets.data(), count);
  }
  std::vector<std::string> strings;
  strings.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    size_t end = i + 1 < count ? offsets[i + 1] : length;
    strings.emplace_back(buffer, offsets[i], end - offsets[i]);
  }
  return strings;
}

nb::object ViewArray(void* data, const std::vector<int64_t>& dims, ONNXTensorElementDataType element_type,
                     nb::handle owner) {
  std::vector<size_t> shape(dims.begin(), dims.end());
  return nb::ndarray<nb::numpy>(data, shape.size(), shape.data(), owner, nullptr, DlpackType(element_type), kDLCPU)
      .cast();
}

nb::object DescribeType(Ort::ConstTypeInfo type_info) {
  switch (type_info.GetONNXType()) {
    case ONNX_TYPE_TENSOR:
    case ONNX_TYPE_SPARSETENSOR: {
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      nb::object shape = nb::none();
      if (tensor_info.HasShape()) {
        std::vector<int64_t> dims = tensor_info.GetShape();
        std::vector<const char*> names = tensor_info.GetSymbolicDimensions();
        nb::list dims_list;
        for (size_t i = 0; i < dims.size(); ++i) {
          if (dims[i] >= 0) {
            dims_list.append(dims[i]);
          } else if (names[i] != nullptr && names[i][0] != '\0') {
            dims_list.append(nb::str(names[i]));
          } else {
            dims_list.append(nb::none());
          }
        }
        shape = dims_list;
      }
      const char* kind = type_info.GetONNXType() == ONNX_TYPE_TENSOR ? "tensor" : "sparse_tensor";
      return nb::make_tuple(kind, static_cast<int>(tensor_info.GetElementType()), shape);
    }
    case ONNX_TYPE_SEQUENCE:
      return nb::make_tuple("seq", DescribeType(type_info.GetSequenceTypeInfo().GetSequenceElementType().GetConst()));
    case ONNX_TYPE_MAP: {
      auto map_info = type_info.GetMapTypeInfo();
      return nb::make_tuple("map", static_cast<int>(map_info.GetMapKeyType()),
                            DescribeType(map_info.GetMapValueType().GetConst()));
    }
    case ONNX_TYPE_OPTIONAL:
      return nb::make_tuple("optional",
                            DescribeType(type_info.GetOptionalTypeInfo().GetOptionalElementType().GetConst()));
    case ONNX_TYPE_OPAQUE:
      return nb::make_tuple("opaque");
    default:
      return nb::make_tuple("unknown");
  }
}

void BindValue(nb::module_& m) {
  nb::class_<MemoryInfo>(m, "OrtMemoryInfo")
      .def(
          "__init__",
          [](MemoryInfo* self, const std::string& name, int allocator_type, int device_id, int mem_type) {
            new (self) MemoryInfo(Ort::MemoryInfo(name.c_str(), static_cast<OrtAllocatorType>(allocator_type),
                                                  device_id, static_cast<OrtMemType>(mem_type)));
          },
          nb::arg("name"), nb::arg("allocator_type"), nb::arg("device_id"), nb::arg("mem_type"))
      .def_static(
          "create_v2",
          [](const std::string& name, int device_type, uint32_t vendor_id, int32_t device_id, int device_mem_type,
             size_t alignment, int allocator_type) {
            return MemoryInfo(Ort::MemoryInfo(name.c_str(), static_cast<OrtMemoryInfoDeviceType>(device_type),
                                              vendor_id, device_id, static_cast<OrtDeviceMemoryType>(device_mem_type),
                                              alignment, static_cast<OrtAllocatorType>(allocator_type)));
          },
          nb::arg("name"), nb::arg("device_type"), nb::arg("vendor_id"), nb::arg("device_id"),
          nb::arg("device_mem_type"), nb::arg("alignment"), nb::arg("allocator_type"))
      .def_prop_ro("name", [](const MemoryInfo& i) { return i.Get().GetAllocatorName(); })
      .def_prop_ro("device_id", [](const MemoryInfo& i) { return i.Get().GetDeviceId(); })
      .def_prop_ro("mem_type", [](const MemoryInfo& i) { return PyEnum("OrtMemType", i.Get().GetMemoryType()); })
      .def_prop_ro("allocator_type", [](const MemoryInfo& i) { return PyEnum("OrtAllocatorType", i.Get().GetAllocatorType()); })
      .def_prop_ro("device_type",
                   [](const MemoryInfo& i) { return PyEnum("OrtMemoryInfoDeviceType", i.Get().GetDeviceType()); })
      .def_prop_ro("device_mem_type",
                   [](const MemoryInfo& i) { return PyEnum("OrtDeviceMemoryType", i.Get().GetDeviceMemoryType()); })
      .def_prop_ro("device_vendor_id", [](const MemoryInfo& i) { return i.Get().GetVendorId(); })
      .def("__eq__", [](const MemoryInfo& a, const MemoryInfo& b) { return a.Get() == b.Get(); });

  nb::class_<Value>(m, "OrtValue")
      .def_static(
          "from_array",
          [](nb::handle array, std::optional<int> element_type) {
            return TensorFromArray(nb::cast<AnyArray>(array), array, element_type);
          },
          nb::arg("array"), nb::arg("element_type").none() = nb::none(),
          "Return a tensor that borrows the buffer of the C-contiguous `array`, which supports DLPack or the buffer "
          "protocol; `element_type` reinterprets the buffer.")
      .def_static("from_strings", &TensorFromStrings, nb::arg("strings"), nb::arg("shape"))
      .def_static(
          "from_pointer",
          [](uintptr_t data, const std::vector<int64_t>& shape, int element_type, const MemoryInfo& memory_info) {
            OrtValue* out = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateTensorWithDataAsOrtValue(
                memory_info.ptr, reinterpret_cast<void*>(data), std::numeric_limits<size_t>::max(), shape.data(),
                shape.size(), static_cast<ONNXTensorElementDataType>(element_type), &out));
            return Value{{}, Ort::Value(out)};
          },
          nb::arg("data"), nb::arg("shape"), nb::arg("element_type"), nb::arg("memory_info"),
          "Return a tensor that borrows the buffer at the address `data`, which must outlive it.")
      .def_static(
          "from_dlpack",
          [](nb::handle data) {
            AnyArray array;
            if (!nb::try_cast(data, array, false)) {
              throw nb::type_error(
                  "from_dlpack expected a valid DLPack capsule named 'dltensor' or an object with '__dlpack__'");
            }
            // The imported array keeps the buffer of the producer alive.
            nb::capsule owner(new AnyArray(array), [](void* p) noexcept { delete static_cast<AnyArray*>(p); });
            return TensorFromArray(array, owner, std::nullopt);
          },
          nb::arg("data").none())
      .def_static(
          "empty",
          [](const std::vector<int64_t>& shape, int element_type, const MemoryInfo* memory_info) {
            auto cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
            OrtAllocator* allocator = GetAllocator(memory_info ? memory_info->ptr : cpu);
            return Value{{},
                         Ort::Value::CreateTensor(allocator, shape.data(), shape.size(),
                                                  static_cast<ONNXTensorElementDataType>(element_type))};
          },
          nb::arg("shape"), nb::arg("element_type"), nb::arg("memory_info").none(),
          "Return an uninitialized tensor allocated by the default allocator of `memory_info`; on the CPU if None.")
      .def_static(
          "sequence",
          [](const std::vector<const Value*>& elements) {
            std::vector<const OrtValue*> values;
            nb::list owners;
            for (const Value* element : elements) {
              values.push_back(element->value);
              if (element->owner.is_valid()) {
                owners.append(element->owner);
              }
            }
            OrtValue* sequence = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateValue(values.data(), values.size(), ONNX_TYPE_SEQUENCE, &sequence));
            return Value{owners, Ort::Value(sequence)};
          },
          nb::arg("elements"), "Return a sequence of the tensors or maps `elements`; it may not be empty.")
      .def_static(
          "map",
          [](const Value& keys, const Value& values) {
            return Value{{}, Ort::Value::CreateMap(keys.value, values.value)};
          },
          nb::arg("keys"), nb::arg("values"), "Return a map of the elements of the 1-D tensors `keys` and `values`.")
      .def_static(
          "sparse_coo",
          [](const std::vector<int64_t>& dense_shape, nb::handle values_obj, nb::handle indices_obj) {
            const Value& values = Unwrap(values_obj);
            const Value& indices = Unwrap(indices_obj);
            return SparseTensor(
                dense_shape, values, nb::make_tuple(values_obj, indices_obj),
                [&](Ort::Value& v) { v.UseCooIndices(MutableData(indices), ElementCount(indices)); },
                [&](Ort::Value& v, auto& info, const auto& param) {
                  v.FillSparseTensorCoo(info, param, indices.value.GetTensorData<int64_t>(),
                                        ElementCount(indices));
                });
          },
          nb::arg("dense_shape"), nb::arg("values"), nb::arg("indices"))
      .def_static(
          "sparse_csr",
          [](const std::vector<int64_t>& dense_shape, nb::handle values_obj, nb::handle inner_obj, nb::handle outer_obj) {
            const Value& values = Unwrap(values_obj);
            const Value& inner = Unwrap(inner_obj);
            const Value& outer = Unwrap(outer_obj);
            return SparseTensor(
                dense_shape, values, nb::make_tuple(values_obj, inner_obj, outer_obj),
                [&](Ort::Value& v) {
                  v.UseCsrIndices(MutableData(inner), ElementCount(inner), MutableData(outer), ElementCount(outer));
                },
                [&](Ort::Value& v, auto& info, const auto& param) {
                  v.FillSparseTensorCsr(info, param, inner.value.GetTensorData<int64_t>(),
                                        ElementCount(inner), outer.value.GetTensorData<int64_t>(),
                                        ElementCount(outer));
                });
          },
          nb::arg("dense_shape"), nb::arg("values"), nb::arg("inner_indices"), nb::arg("outer_indices"))
      .def_prop_ro(
          "onnx_type", [](const Value& v) { return static_cast<int>(v.value.GetTypeInfo().GetONNXType()); },
          "The `ONNXType` of the value.")
      .def_prop_ro("type", [](const Value& v) { return DescribeType(v.value.GetTypeInfo().GetConst()); })
      .def_prop_ro("has_value", [](const Value& v) { return v.value.HasValue(); })
      .def_prop_ro("element_type",
                   [](const Value& v) { return static_cast<int>(v.value.GetTensorTypeAndShapeInfo().GetElementType()); })
      .def_prop_ro("shape", [](const Value& v) { return v.value.GetTensorTypeAndShapeInfo().GetShape(); })
      .def_prop_ro("size_in_bytes", [](const Value& v) { return v.value.GetTensorSizeInBytes(); })
      .def_prop_ro("data_ptr", [](Value& v) { return reinterpret_cast<uintptr_t>(v.value.GetTensorMutableRawData()); })
      .def_prop_ro("memory_info",
                   [](nb::handle self) { return MemoryInfo(Unwrap(self).value.GetTensorMemoryInfo(), nb::borrow(self)); })
      .def_prop_ro(
          "count", [](const Value& v) { return v.value.GetCount(); },
          "The number of elements of a sequence, or 2 for a map.")
      .def(
          "element",
          [](const Value& v, int index) {
            Ort::AllocatorWithDefaultOptions allocator;
            return Value{v.owner, v.value.GetValue(index, allocator)};
          },
          nb::arg("index"), "Return the `index`-th element of a sequence, or the keys (0) or values (1) of a map.")
      .def(
          "numpy", [](nb::handle self) { return TensorToArray(self, true); },
          "Return a numpy array that views the buffer of the numeric tensor and keeps this value alive.")
      .def("strings",
           [](const Value& v) { return GetStrings(v.value, v.value.GetTensorTypeAndShapeInfo().GetElementCount()); })
      .def(
          "__dlpack__", [](nb::handle self, nb::kwargs) { return TensorToArray(self, false); },
          "Return a DLPack capsule that views the buffer of the tensor and keeps this value alive.")
      .def("__dlpack_device__", [](const Value& v) { return DlpackDevice(v.value.GetTensorMemoryInfo()); })
      .def(
          "copy_from",
          [](Value& dst, const Value& src) {
            const OrtValue* src_ptr = src.value;
            OrtValue* dst_ptr = dst.value;
            Ort::ThrowOnError(Ort::GetApi().CopyTensors(GetEnv(), &src_ptr, &dst_ptr, nullptr, 1));
          },
          nb::arg("src"), "Copy the data of the tensor `src`, which has the same type and shape, into this tensor.")
      .def_prop_ro("sparse_format", [](const Value& v) { return static_cast<int>(v.value.GetSparseFormat()); })
      .def_prop_ro("sparse_values_type",
                   [](const Value& v) {
                     auto info = v.value.GetSparseTensorValuesTypeAndShapeInfo();
                     return std::make_pair(static_cast<int>(info.GetElementType()), info.GetShape());
                   })
      .def(
          "sparse_values",
          [](nb::handle self) -> nb::object {
            Ort::Value& v = Unwrap(self).value;
            auto info = v.GetSparseTensorValuesTypeAndShapeInfo();
            if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
              return nb::cast(GetStrings(v, info.GetElementCount()));
            }
            return ViewArray(const_cast<void*>(v.GetSparseTensorValues<void>()), info.GetShape(),
                             info.GetElementType(), self);
          },
          "Return a numpy array that views the numeric values, or a list of the string values.")
      .def(
          "sparse_indices",
          [](nb::handle self, int format) {
            Ort::Value& v = Unwrap(self).value;
            auto indices_format = static_cast<OrtSparseIndicesFormat>(format);
            auto info = v.GetSparseTensorIndicesTypeShapeInfo(indices_format);
            size_t count = 0;
            const void* data = v.GetSparseTensorIndicesData<void>(indices_format, count);
            return ViewArray(const_cast<void*>(data), info.GetShape(), info.GetElementType(), self);
          },
          nb::arg("format"), "Return a numpy array that views the indices of `OrtSparseIndicesFormat` `format`.");
}

}  // namespace onnxruntime_nb
