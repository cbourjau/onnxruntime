// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Bindings of hardware devices and the execution provider devices that the environment owns.

#pragma once

#include <map>
#include <string>

#include <nanobind/nanobind.h>

#include "onnxruntime_cxx_api.h"

namespace onnxruntime_nb {

struct HardwareDevice {
  const OrtHardwareDevice* ptr;
};

struct EpDevice {
  const OrtEpDevice* ptr;
};

std::map<std::string, std::string> ToDict(const OrtKeyValuePairs* pairs);

void BindDevices(nanobind::module_& m);

}  // namespace onnxruntime_nb
