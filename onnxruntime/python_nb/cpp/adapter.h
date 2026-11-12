// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Serialization of LoRA adapters in the ONNX Runtime adapter file format.

#pragma once

#include <nanobind/nanobind.h>

namespace onnxruntime_nb {

void BindAdapterFormat(nanobind::module_& m);

}  // namespace onnxruntime_nb
