// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Conversion between tensors and Python objects.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "ort_api.h"

namespace onnxruntime_nb {

namespace nb = nanobind;

// Read-only so that non-writeable numpy arrays are accepted; ORT does not write to inputs.
using InputArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

// Flattened elements and shape of a string tensor.
using StringTensor = std::pair<std::vector<std::string>, std::vector<int64_t>>;

// Returns a tensor that borrows the buffer of `array`; `name` is used in error messages.
OrtPtr<OrtValue> CreateTensorView(const InputArray& array, const OrtMemoryInfo* memory_info,
                                  const std::string& name);

// Returns a new tensor holding a copy of the strings.
OrtPtr<OrtValue> CreateStringTensor(const StringTensor& tensor);

// Returns a numpy array that views the tensor's buffer and owns `value`, or a `StringTensor` copy for strings.
nb::object ToPython(OrtPtr<OrtValue> value);

}  // namespace onnxruntime_nb
