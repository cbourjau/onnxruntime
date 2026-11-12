// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Conversion between OrtValues and Python objects.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "onnxruntime_cxx_api.h"

namespace onnxruntime_nb {

namespace nb = nanobind;

// An OrtValue together with the Python object that owns its buffer, if the buffer is borrowed.
struct Value {
  // Keeps alive what `value` refers to, e.g. its buffer or allocator; declared first to outlive `value`.
  nb::object owner;
  Ort::Value value{nullptr};
};

// An OrtMemoryInfo that is either owned or borrowed from `owner`.
struct MemoryInfo {
  Ort::MemoryInfo owned{nullptr};
  const OrtMemoryInfo* ptr = nullptr;
  nb::object owner;

  explicit MemoryInfo(Ort::MemoryInfo info) : owned(std::move(info)), ptr(owned) {}
  MemoryInfo(const OrtMemoryInfo* info, nb::object info_owner) : ptr(info), owner(std::move(info_owner)) {}
  Ort::ConstMemoryInfo Get() const { return Ort::ConstMemoryInfo{ptr}; }
};

// Returns the process-wide environment, creating it on first use.
Ort::Env& GetEnv();

// Returns the default allocator for `memory_info`: the default CPU allocator or a shared allocator of the environment.
OrtAllocator* GetAllocator(const OrtMemoryInfo* memory_info);

// Returns a copy of the elements of the string tensor or sparse tensor `value` in row-major order.
std::vector<std::string> GetStrings(const OrtValue* value, size_t count);

// Returns a numpy array that views `count` elements of `element_type` at `data`, owned by `owner`.
nb::object ViewArray(void* data, const std::vector<int64_t>& shape, ONNXTensorElementDataType element_type,
                     nb::handle owner);

// Returns a nested tuple describing `type_info`, e.g. ('map', 7, ('tensor', 1, None)).
nb::object DescribeType(Ort::ConstTypeInfo type_info);

// Returns the member with value `value` of the enumeration `name` of `onnxruntime_nb._types`.
inline nb::object PyEnum(const char* name, int value) {
  return nb::module_::import_("onnxruntime_nb._types").attr(name)(value);
}

void BindValue(nb::module_& m);

}  // namespace onnxruntime_nb
