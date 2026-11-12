// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "adapter.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "lora/adapter_format/adapter_schema.fbs.h"
#include "value.h"

namespace onnxruntime_nb {

namespace {

namespace fbs = onnxruntime::adapters;

// Version of the adapter file format; see onnxruntime/lora/adapter_format_version.h.
constexpr int kAdapterFormatVersion = 1;

// Returns the serialized adapter with the tensors `parameters`, given as (name, value) pairs.
nb::bytes Serialize(int adapter_version, int model_version, const std::vector<std::pair<std::string, const Value*>>& parameters) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<fbs::Parameter>> offsets;
  for (const auto& [name, value] : parameters) {
    if (!value->value.IsTensor()) {
      throw Ort::Exception("Lora adapter parameter '" + name + "' must be a Tensor OrtValue.", ORT_INVALID_ARGUMENT);
    }
    auto info = value->value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      throw Ort::Exception("Lora adapter parameter '" + name +
                               "' has element type STRING, which is not supported by the adapter format.",
                           ORT_INVALID_ARGUMENT);
    }
    if (value->value.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
      throw Ort::Exception("Lora adapter parameter '" + name + "' must reside on CPU to be exported.",
                           ORT_INVALID_ARGUMENT);
    }
    std::vector<int64_t> dims = info.GetShape();
    builder.ForceVectorAlignment(value->value.GetTensorSizeInBytes(), sizeof(uint8_t), 8);
    auto data = builder.CreateVector(static_cast<const uint8_t*>(value->value.GetTensorRawData()),
                                     value->value.GetTensorSizeInBytes());
    offsets.push_back(fbs::CreateParameter(builder, builder.CreateString(name), builder.CreateVector(dims),
                                           static_cast<fbs::TensorDataType>(info.GetElementType()), data));
  }
  auto adapter = fbs::CreateAdapter(builder, kAdapterFormatVersion, adapter_version, model_version,
                                    builder.CreateVector(offsets));
  fbs::FinishAdapterBuffer(builder, adapter);
  return nb::bytes(builder.GetBufferPointer(), builder.GetSize());
}

// Returns (format version, adapter version, model version, [(name, value)]); the values view `data`.
nb::tuple Deserialize(nb::bytes data) {
  auto bytes = reinterpret_cast<const uint8_t*>(data.c_str());
  flatbuffers::Verifier verifier(bytes, data.size());
  if (!fbs::AdapterBufferHasIdentifier(bytes) || !fbs::VerifyAdapterBuffer(verifier)) {
    throw Ort::Exception("the data is not a valid adapter", ORT_INVALID_ARGUMENT);
  }
  const fbs::Adapter* adapter = fbs::GetAdapter(bytes);
  if (adapter->format_version() != kAdapterFormatVersion) {
    throw Ort::Exception("unsupported adapter format version " + std::to_string(adapter->format_version()),
                         ORT_INVALID_ARGUMENT);
  }
  auto cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  nb::list parameters;
  if (adapter->parameters() != nullptr) {
    for (const fbs::Parameter* p : *adapter->parameters()) {
      std::vector<int64_t> dims(p->dims()->begin(), p->dims()->end());
      auto value = Ort::Value::CreateTensor(cpu, const_cast<uint8_t*>(p->raw_data()->data()), p->raw_data()->size(),
                                            dims.data(), dims.size(),
                                            static_cast<ONNXTensorElementDataType>(p->data_type()));
      parameters.append(nb::make_tuple(p->name()->str(), Value{data, std::move(value)}));
    }
  }
  return nb::make_tuple(adapter->format_version(), adapter->adapter_version(), adapter->model_version(), parameters);
}

}  // namespace

void BindAdapterFormat(nb::module_& m) {
  m.def("serialize_adapter", &Serialize, nb::arg("adapter_version"), nb::arg("model_version"), nb::arg("parameters"),
        "Return the adapter file content with the CPU tensors `parameters`, given as (name, value) pairs.");
  m.def("deserialize_adapter", &Deserialize, nb::arg("data"),
        "Return (format version, adapter version, model version, [(name, value)]) of the adapter file content "
        "`data`; the values view `data`.");
}

}  // namespace onnxruntime_nb
