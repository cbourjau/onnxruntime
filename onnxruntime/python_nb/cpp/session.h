// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Bindings of sessions and their options.

#pragma once

#include <nanobind/nanobind.h>

namespace onnxruntime_nb {

void BindSession(nanobind::module_& m);

}  // namespace onnxruntime_nb
