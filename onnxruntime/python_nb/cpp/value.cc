// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "value.h"

#include <optional>
#include <utility>

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

}  // namespace

Ort::Value TensorFromArray(const InputArray& array) {
  std::optional<ONNXTensorElementDataType> element_type;
  for (const auto& [onnx_type, dl_type] : kElementTypes) {
    if (dl_type == array.dtype()) {
      element_type = onnx_type;
    }
  }
  if (!element_type) {
    throw Ort::Exception("arrays with dtype code " + std::to_string(array.dtype().code) + " and " +
                             std::to_string(array.dtype().bits) + " bits are not supported",
                         ORT_INVALID_ARGUMENT);
  }
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> shape(array.shape_ptr(), array.shape_ptr() + array.ndim());
  return Ort::Value::CreateTensor(memory_info, const_cast<void*>(array.data()), array.nbytes(), shape.data(),
                                  shape.size(), *element_type);
}

Ort::Value TensorFromStrings(const std::vector<std::string>& strings, const std::vector<int64_t>& shape) {
  std::vector<const char*> pointers;
  for (const auto& s : strings) {
    pointers.push_back(s.c_str());
  }
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Value value =
      Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
  value.FillStringTensor(pointers.data(), pointers.size());
  return value;
}

nb::object TensorToArray(Ort::Value& value, nb::handle owner) {
  auto info = value.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType element_type = info.GetElementType();
  std::optional<nb::dlpack::dtype> dtype;
  for (const auto& [onnx_type, dl_type] : kElementTypes) {
    if (onnx_type == element_type) {
      dtype = dl_type;
    }
  }
  if (!dtype) {
    throw Ort::Exception("tensors of element type " + std::to_string(element_type) + " are not supported",
                         ORT_NOT_IMPLEMENTED);
  }
  std::vector<int64_t> dims = info.GetShape();
  std::vector<size_t> shape(dims.begin(), dims.end());
  return OutputArray(value.GetTensorMutableRawData(), shape.size(), shape.data(), owner, nullptr, *dtype).cast();
}

std::vector<std::string> TensorToStrings(const Ort::Value& value) {
  size_t count = value.GetTensorTypeAndShapeInfo().GetElementCount();
  size_t length = value.GetStringTensorDataLength();
  std::string buffer(length, '\0');
  std::vector<size_t> offsets(count);
  value.GetStringTensorContent(buffer.data(), length, offsets.data(), count);

  std::vector<std::string> strings;
  for (size_t i = 0; i < count; ++i) {
    size_t end = i + 1 < count ? offsets[i + 1] : length;
    strings.emplace_back(buffer, offsets[i], end - offsets[i]);
  }
  return strings;
}

}  // namespace onnxruntime_nb
