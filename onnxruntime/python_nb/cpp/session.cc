// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session.h"

#include <utility>

namespace onnxruntime_nb {

namespace {

using CountFn = OrtStatus*(ORT_API_CALL*)(const OrtSession*, size_t*);
using NameFn = OrtStatus*(ORT_API_CALL*)(const OrtSession*, size_t, OrtAllocator*, char**);

std::vector<std::string> GetNames(const OrtSession* session, CountFn count_fn, NameFn name_fn) {
  OrtAllocator* allocator = GetDefaultAllocator();
  size_t count = 0;
  ThrowOnError(count_fn(session, &count));
  std::vector<std::string> names;
  for (size_t i = 0; i < count; ++i) {
    char* name = nullptr;
    ThrowOnError(name_fn(session, i, allocator, &name));
    names.emplace_back(name);
    ThrowOnError(g_ort->AllocatorFree(allocator, name));
  }
  return names;
}

}  // namespace

// SessionOptions
// --------------

SessionOptions::SessionOptions() {
  OrtSessionOptions* p = nullptr;
  ThrowOnError(g_ort->CreateSessionOptions(&p));
  options_.reset(p);
}

void SessionOptions::SetIntraOpNumThreads(int n) { ThrowOnError(g_ort->SetIntraOpNumThreads(options_.get(), n)); }

void SessionOptions::SetInterOpNumThreads(int n) { ThrowOnError(g_ort->SetInterOpNumThreads(options_.get(), n)); }

void SessionOptions::SetGraphOptimizationLevel(int level) {
  ThrowOnError(g_ort->SetSessionGraphOptimizationLevel(options_.get(), static_cast<GraphOptimizationLevel>(level)));
}

void SessionOptions::SetExecutionMode(int mode) {
  ThrowOnError(g_ort->SetSessionExecutionMode(options_.get(), static_cast<ExecutionMode>(mode)));
}

void SessionOptions::SetLogSeverityLevel(int level) {
  ThrowOnError(g_ort->SetSessionLogSeverityLevel(options_.get(), level));
}

void SessionOptions::SetLogVerbosityLevel(int level) {
  ThrowOnError(g_ort->SetSessionLogVerbosityLevel(options_.get(), level));
}

void SessionOptions::SetLogId(const std::string& logid) {
  ThrowOnError(g_ort->SetSessionLogId(options_.get(), logid.c_str()));
}

void SessionOptions::EnableProfiling(const std::filesystem::path& prefix) {
  ThrowOnError(g_ort->EnableProfiling(options_.get(), prefix.c_str()));
}

void SessionOptions::SetOptimizedModelFilePath(const std::filesystem::path& path) {
  ThrowOnError(g_ort->SetOptimizedModelFilePath(options_.get(), path.c_str()));
}

void SessionOptions::SetCpuMemArena(bool enable) {
  ThrowOnError(enable ? g_ort->EnableCpuMemArena(options_.get()) : g_ort->DisableCpuMemArena(options_.get()));
}

void SessionOptions::SetMemPattern(bool enable) {
  ThrowOnError(enable ? g_ort->EnableMemPattern(options_.get()) : g_ort->DisableMemPattern(options_.get()));
}

void SessionOptions::AddConfigEntry(const std::string& key, const std::string& value) {
  ThrowOnError(g_ort->AddSessionConfigEntry(options_.get(), key.c_str(), value.c_str()));
}

void SessionOptions::AppendExecutionProvider(const std::string& name,
                                             const std::map<std::string, std::string>& options) {
  std::vector<const char*> keys;
  std::vector<const char*> values;
  for (const auto& [key, value] : options) {
    keys.push_back(key.c_str());
    values.push_back(value.c_str());
  }
  ThrowOnError(g_ort->SessionOptionsAppendExecutionProvider(options_.get(), name.c_str(), keys.data(), values.data(),
                                                            keys.size()));
}

// Session
// -------

Session::Session(const std::filesystem::path& model_path, const SessionOptions& options) : env_(GetEnv()) {
  OrtSession* session = nullptr;
  ThrowOnError(g_ort->CreateSession(env_.get(), model_path.c_str(), options.get(), &session));
  session_.reset(session);
  input_names_ = GetNames(session, g_ort->SessionGetInputCount, g_ort->SessionGetInputName);
  output_names_ = GetNames(session, g_ort->SessionGetOutputCount, g_ort->SessionGetOutputName);
}

std::vector<nb::object> Session::Run(const std::vector<std::string>& output_names,
                                     const std::map<std::string, InputArray>& arrays,
                                     const std::map<std::string, StringTensor>& strings) {
  OrtMemoryInfo* memory_info = nullptr;
  ThrowOnError(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  OrtPtr<OrtMemoryInfo> memory_info_owner(memory_info);

  std::vector<OrtPtr<OrtValue>> inputs;
  std::vector<const char*> input_names;
  for (const auto& [name, array] : arrays) {
    inputs.push_back(CreateTensorView(array, memory_info, name));
    input_names.push_back(name.c_str());
  }
  for (const auto& [name, tensor] : strings) {
    inputs.push_back(CreateStringTensor(tensor));
    input_names.push_back(name.c_str());
  }
  std::vector<const OrtValue*> input_values;
  for (const auto& input : inputs) {
    input_values.push_back(input.get());
  }
  std::vector<const char*> output_name_ptrs;
  for (const auto& name : output_names) {
    output_name_ptrs.push_back(name.c_str());
  }
  std::vector<OrtValue*> outputs(output_names.size(), nullptr);

  OrtStatus* status = nullptr;
  {
    nb::gil_scoped_release release;
    status = g_ort->Run(session_.get(), nullptr, input_names.data(), input_values.data(), input_values.size(),
                        output_name_ptrs.data(), output_name_ptrs.size(), outputs.data());
  }
  ThrowOnError(status);

  std::vector<OrtPtr<OrtValue>> output_owners;
  for (OrtValue* value : outputs) {
    output_owners.emplace_back(value);
  }
  std::vector<nb::object> result;
  for (auto& value : output_owners) {
    result.push_back(ToPython(std::move(value)));
  }
  return result;
}

}  // namespace onnxruntime_nb
