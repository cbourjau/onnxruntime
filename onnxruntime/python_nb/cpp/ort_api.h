// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Access to the C API: error handling, ownership of C API objects, and process-wide objects.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

namespace onnxruntime_nb {

// The C API function table; set by `InitApi`.
extern const OrtApi* g_ort;

// Sets `g_ort`; throws if the loaded library does not support the API version the module was compiled against.
void InitApi();

class OrtError : public std::runtime_error {
 public:
  OrtError(OrtErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}
  OrtErrorCode code() const { return code_; }

 private:
  OrtErrorCode code_;
};

// Releases `status` and throws an `OrtError` if it is an error.
void ThrowOnError(OrtStatus* status);

struct OrtDeleter {
  void operator()(OrtEnv* p) const { g_ort->ReleaseEnv(p); }
  void operator()(OrtMemoryInfo* p) const { g_ort->ReleaseMemoryInfo(p); }
  void operator()(OrtSession* p) const { g_ort->ReleaseSession(p); }
  void operator()(OrtSessionOptions* p) const { g_ort->ReleaseSessionOptions(p); }
  void operator()(OrtTensorTypeAndShapeInfo* p) const { g_ort->ReleaseTensorTypeAndShapeInfo(p); }
  void operator()(OrtValue* p) const { g_ort->ReleaseValue(p); }
};

template <typename T>
using OrtPtr = std::unique_ptr<T, OrtDeleter>;

// The process-wide environment, created on first use. Holders keep it alive.
std::shared_ptr<OrtEnv> GetEnv();

OrtAllocator* GetDefaultAllocator();

std::vector<std::string> GetAvailableProviders();

}  // namespace onnxruntime_nb
