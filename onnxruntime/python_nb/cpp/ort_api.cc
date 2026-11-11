// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_api.h"

namespace onnxruntime_nb {

const OrtApi* g_ort = nullptr;

void InitApi() {
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (g_ort == nullptr) {
    throw std::runtime_error("the loaded onnxruntime library does not support ORT_API_VERSION " +
                             std::to_string(ORT_API_VERSION));
  }
}

void ThrowOnError(OrtStatus* status) {
  if (status == nullptr) {
    return;
  }
  OrtErrorCode code = g_ort->GetErrorCode(status);
  std::string message = g_ort->GetErrorMessage(status);
  g_ort->ReleaseStatus(status);
  throw OrtError(code, message);
}

std::shared_ptr<OrtEnv> GetEnv() {
  static std::shared_ptr<OrtEnv> env = [] {
    OrtEnv* p = nullptr;
    ThrowOnError(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_nb", &p));
    return std::shared_ptr<OrtEnv>(p, OrtDeleter{});
  }();
  return env;
}

OrtAllocator* GetDefaultAllocator() {
  OrtAllocator* allocator = nullptr;
  ThrowOnError(g_ort->GetAllocatorWithDefaultOptions(&allocator));
  return allocator;
}

std::vector<std::string> GetAvailableProviders() {
  char** providers = nullptr;
  int length = 0;
  ThrowOnError(g_ort->GetAvailableProviders(&providers, &length));
  std::vector<std::string> result(providers, providers + length);
  ThrowOnError(g_ort->ReleaseAvailableProviders(providers, length));
  return result;
}

}  // namespace onnxruntime_nb
