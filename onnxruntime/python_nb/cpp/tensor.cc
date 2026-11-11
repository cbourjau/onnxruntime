// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensor.h"

#include <optional>

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace onnxruntime_nb {

namespace {

using nb::dlpack::dtype_code;

const std::pair<ONNXTensorElementDataType, nb::dlpack::dtype> kElementTypes[] = {
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, nb::dtype<float>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, nb::dtype<double>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {static_cast<uint8_t>(dtype_code::Float), 16, 1}},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, {static_cast<uint8_t>(dtype_code::Bfloat), 16, 1}},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, nb::dtype<int8_t>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, nb::dtype<int16_t>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, nb::dtype<int32_t>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, nb::dtype<int64_t>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, nb::dtype<uint8_t>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, nb::dtype<uint16_t>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, nb::dtype<uint32_t>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, nb::dtype<uint64_t>()},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, nb::dtype<bool>()},
};

using OutputArray = nb::ndarray<nb::numpy>;

StringTensor ReadStringTensor(const OrtValue* value, std::vector<int64_t> shape) {
  size_t count = 1;
  for (int64_t dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  size_t length = 0;
  ThrowOnError(g_ort->GetStringTensorDataLength(value, &length));
  std::string buffer(length, '\0');
  std::vector<size_t> offsets(count);
  ThrowOnError(g_ort->GetStringTensorContent(value, buffer.data(), length, offsets.data(), count));

  std::vector<std::string> strings;
  for (size_t i = 0; i < count; ++i) {
    size_t end = i + 1 < count ? offsets[i + 1] : length;
    strings.emplace_back(buffer, offsets[i], end - offsets[i]);
  }
  return {std::move(strings), std::move(shape)};
}

}  // namespace

OrtPtr<OrtValue> CreateTensorView(const InputArray& array, const OrtMemoryInfo* memory_info,
                                  const std::string& name) {
  std::optional<ONNXTensorElementDataType> element_type;
  for (const auto& [onnx_type, dl_type] : kElementTypes) {
    if (dl_type == array.dtype()) {
      element_type = onnx_type;
    }
  }
  if (!element_type) {
    throw OrtError(ORT_INVALID_ARGUMENT, "input `" + name + "` has an unsupported dtype");
  }
  std::vector<int64_t> shape(array.shape_ptr(), array.shape_ptr() + array.ndim());
  OrtValue* value = nullptr;
  ThrowOnError(g_ort->CreateTensorWithDataAsOrtValue(memory_info, const_cast<void*>(array.data()), array.nbytes(),
                                                     shape.data(), shape.size(), *element_type, &value));
  return OrtPtr<OrtValue>(value);
}

OrtPtr<OrtValue> CreateStringTensor(const StringTensor& tensor) {
  const auto& [strings, shape] = tensor;
  std::vector<const char*> pointers;
  for (const auto& s : strings) {
    pointers.push_back(s.c_str());
  }
  OrtValue* value = nullptr;
  ThrowOnError(g_ort->CreateTensorAsOrtValue(GetDefaultAllocator(), shape.data(), shape.size(),
                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &value));
  OrtPtr<OrtValue> owner(value);
  ThrowOnError(g_ort->FillStringTensor(value, pointers.data(), pointers.size()));
  return owner;
}

nb::object ToPython(OrtPtr<OrtValue> value) {
  OrtTensorTypeAndShapeInfo* info = nullptr;
  ThrowOnError(g_ort->GetTensorTypeAndShape(value.get(), &info));
  OrtPtr<OrtTensorTypeAndShapeInfo> info_owner(info);
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ThrowOnError(g_ort->GetTensorElementType(info, &element_type));
  size_t ndim = 0;
  ThrowOnError(g_ort->GetDimensionsCount(info, &ndim));
  std::vector<int64_t> dims(ndim);
  ThrowOnError(g_ort->GetDimensions(info, dims.data(), ndim));

  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    return nb::cast(ReadStringTensor(value.get(), std::move(dims)));
  }
  std::optional<nb::dlpack::dtype> dtype;
  for (const auto& [onnx_type, dl_type] : kElementTypes) {
    if (onnx_type == element_type) {
      dtype = dl_type;
    }
  }
  if (!dtype) {
    throw OrtError(ORT_NOT_IMPLEMENTED,
                   "outputs of element type " + std::to_string(element_type) + " are not supported");
  }
  std::vector<size_t> shape(dims.begin(), dims.end());
  void* data = nullptr;
  ThrowOnError(g_ort->GetTensorMutableData(value.get(), &data));
  nb::capsule owner(value.release(), [](void* p) noexcept { g_ort->ReleaseValue(static_cast<OrtValue*>(p)); });
  return OutputArray(data, shape.size(), shape.data(), owner, nullptr, *dtype).cast();
}

}  // namespace onnxruntime_nb
