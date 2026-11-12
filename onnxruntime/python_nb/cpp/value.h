// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Conversion between tensors and Python objects.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "onnxruntime_cxx_api.h"

namespace onnxruntime_nb {

namespace nb = nanobind;

// Read-only so that non-writeable numpy arrays are accepted; ORT does not write to inputs.
using InputArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

// Returns a tensor that borrows the buffer of `array`; the caller keeps `array` alive.
Ort::Value TensorFromArray(const InputArray& array);

// Returns a new string tensor holding a copy of `strings`, which are given in row-major order.
Ort::Value TensorFromStrings(const std::vector<std::string>& strings, const std::vector<int64_t>& shape);

// Returns a numpy array that views the buffer of the numeric tensor `value`; `owner` keeps `value` alive.
nb::object TensorToArray(Ort::Value& value, nb::handle owner);

// Returns a copy of the elements of the string tensor `value` in row-major order.
std::vector<std::string> TensorToStrings(const Ort::Value& value);

}  // namespace onnxruntime_nb
