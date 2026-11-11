// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Session options and inference sessions.

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>

#include "ort_api.h"
#include "tensor.h"

namespace onnxruntime_nb {

// Owns an `OrtSessionOptions`; each method calls the C API function of the same name.
class SessionOptions {
 public:
  SessionOptions();

  const OrtSessionOptions* get() const { return options_.get(); }

  void SetIntraOpNumThreads(int n);
  void SetInterOpNumThreads(int n);
  void SetGraphOptimizationLevel(int level);
  void SetExecutionMode(int mode);
  void SetLogSeverityLevel(int level);
  void SetLogVerbosityLevel(int level);
  void SetLogId(const std::string& logid);
  void EnableProfiling(const std::filesystem::path& prefix);
  void SetOptimizedModelFilePath(const std::filesystem::path& path);
  void SetCpuMemArena(bool enable);
  void SetMemPattern(bool enable);
  void AddConfigEntry(const std::string& key, const std::string& value);
  void AppendExecutionProvider(const std::string& name, const std::map<std::string, std::string>& options);

 private:
  OrtPtr<OrtSessionOptions> options_;
};

class Session {
 public:
  Session(const std::filesystem::path& model_path, const SessionOptions& options);

  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::string>& OutputNames() const { return output_names_; }

  // Runs the model with the GIL released. The input tensors borrow the buffers of `arrays`,
  // which the caller keeps alive for the duration of the call.
  std::vector<nb::object> Run(const std::vector<std::string>& output_names,
                              const std::map<std::string, InputArray>& arrays,
                              const std::map<std::string, StringTensor>& strings);

 private:
  std::shared_ptr<OrtEnv> env_;
  OrtPtr<OrtSession> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace onnxruntime_nb
