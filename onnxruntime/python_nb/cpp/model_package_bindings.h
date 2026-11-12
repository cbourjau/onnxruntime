// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Bindings of model packages: authoring with the model package library and loading with ONNX Runtime.

#pragma once

#include <nanobind/nanobind.h>

namespace onnxruntime_nb {

void BindModelPackage(nanobind::module_& m);

}  // namespace onnxruntime_nb
