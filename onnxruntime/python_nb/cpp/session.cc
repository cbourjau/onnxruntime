// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session.h"

#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "devices.h"
#include "value.h"

namespace onnxruntime_nb {

namespace {

using StringMap = std::unordered_map<std::string, std::string>;

std::vector<const char*> CStrings(const std::vector<std::string>& strings) {
  std::vector<const char*> pointers;
  pointers.reserve(strings.size());
  for (const auto& s : strings) {
    pointers.push_back(s.c_str());
  }
  return pointers;
}

OrtStatus* ErrorStatus(const char* message) { return Ort::GetApi().CreateStatus(ORT_FAIL, message); }

// Calls `f` with the GIL held and converts exceptions into an OrtStatus.
template <typename F>
OrtStatus* CallPython(F&& f) noexcept {
  nb::gil_scoped_acquire gil;
  try {
    return f();
  } catch (const std::exception& e) {
    return ErrorStatus(e.what());
  }
}

OrtStatus* ORT_API_CALL SelectEpDevices(const OrtEpDevice** ep_devices, size_t num_devices,
                                        const OrtKeyValuePairs* model_metadata,
                                        const OrtKeyValuePairs* runtime_metadata, const OrtEpDevice** selected,
                                        size_t max_selected, size_t* num_selected, void* state) {
  *num_selected = 0;
  return CallPython([&]() -> OrtStatus* {
    nb::list devices;
    for (size_t i = 0; i < num_devices; ++i) {
      devices.append(EpDevice{ep_devices[i]});
    }
    auto* options = static_cast<SessionOptionsState*>(state);
    auto result = nb::cast<std::vector<EpDevice>>(
        options->selection_delegate(devices, ToDict(model_metadata), ToDict(runtime_metadata), max_selected));
    if (result.size() > max_selected) {
      std::string message = "selected too many EP devices (" + std::to_string(result.size()) + "). The limit is " +
                            std::to_string(max_selected) + " EP devices.";
      return ErrorStatus(message.c_str());
    }
    for (size_t i = 0; i < result.size(); ++i) {
      selected[i] = result[i].ptr;
    }
    *num_selected = result.size();
    return nullptr;
  });
}

// The buffer that an EPContext data read callback fills; valid only during the callback.
struct EpContextDataBuffer {
  OrtAllocator* allocator;
  size_t max_size;
  void* data = nullptr;
  size_t size = 0;
  bool allocated = false;
  bool valid = true;

  void CheckValid() const {
    if (!valid) {
      throw std::runtime_error("the EPContext data buffer is only valid during its read callback");
    }
  }
};

OrtStatus* ORT_API_CALL ReadEpContextData(void* state, const char* name, OrtAllocator* allocator, void** buffer,
                                          size_t* data_size) {
  *buffer = nullptr;
  *data_size = 0;
  return CallPython([&]() -> OrtStatus* {
    auto* options = static_cast<SessionOptionsState*>(state);
    auto output = std::make_shared<EpContextDataBuffer>(EpContextDataBuffer{allocator, options->ep_context_data_max_size});
    nb::object py_output = nb::cast(output);
    try {
      options->ep_context_data_read_func(name, py_output);
    } catch (...) {
      output->valid = false;
      if (output->data != nullptr) {
        allocator->Free(allocator, output->data);
      }
      throw;
    }
    output->valid = false;
    if (!output->allocated) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "the EPContext data read callback must allocate its output buffer");
    }
    *buffer = output->data;
    *data_size = output->size;
    return nullptr;
  });
}

// A view of EPContext data that is valid only during its write callback.
struct EpContextData {
  const void* data;
  size_t size;
  bool valid = true;
};

template <typename GetName, typename GetType>
nb::list DescribeArgs(size_t count, GetName get_name, GetType get_type) {
  Ort::AllocatorWithDefaultOptions allocator;
  nb::list args;
  for (size_t i = 0; i < count; ++i) {
    args.append(nb::make_tuple(std::string(get_name(i, allocator).get()), DescribeType(get_type(i).GetConst())));
  }
  return args;
}

std::vector<const OrtValue*> RawValues(const std::vector<const Value*>& values) {
  std::vector<const OrtValue*> raw;
  raw.reserve(values.size());
  for (const Value* value : values) {
    raw.push_back(value->value);
  }
  return raw;
}

const OrtRunOptions* RawRunOptions(const Ort::RunOptions* run_options) {
  return run_options ? static_cast<const OrtRunOptions*>(*run_options) : nullptr;
}

std::vector<Value> Run(Session& s, const std::vector<std::string>& input_names,
                       const std::vector<const Value*>& inputs, const std::vector<std::string>& output_names,
                       const Ort::RunOptions* run_options) {
  auto input_name_ptrs = CStrings(input_names);
  auto output_name_ptrs = CStrings(output_names);
  auto input_values = RawValues(inputs);
  std::vector<OrtValue*> outputs(output_names.size(), nullptr);
  OrtStatus* status = nullptr;
  {
    nb::gil_scoped_release release;
    status = Ort::GetApi().Run(s.session, RawRunOptions(run_options), input_name_ptrs.data(), input_values.data(),
                               input_values.size(), output_name_ptrs.data(), output_name_ptrs.size(), outputs.data());
  }
  Ort::ThrowOnError(status);
  std::vector<Value> result;
  result.reserve(outputs.size());
  for (OrtValue* output : outputs) {
    result.push_back({{}, Ort::Value(output)});
  }
  return result;
}

// The state of an asynchronous run, which must outlive the call.
struct AsyncRun {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<const char*> input_name_ptrs;
  std::vector<const char*> output_name_ptrs;
  std::vector<const OrtValue*> input_values;
  std::vector<OrtValue*> outputs;
  // The session, run options, and inputs must outlive the run.
  nb::object session;
  nb::object run_options;
  nb::object inputs;
  nb::object callback;
};

void RunAsyncDone(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status) {
  std::unique_ptr<AsyncRun> run(static_cast<AsyncRun*>(user_data));
  nb::gil_scoped_acquire gil;
  std::vector<Value> values;
  for (size_t i = 0; i < num_outputs; ++i) {
    values.push_back({{}, Ort::Value(outputs[i])});
  }
  std::string error;
  if (status != nullptr) {
    error = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
  }
  try {
    run->callback(std::move(values), error);
  } catch (nb::python_error& e) {
    e.discard_as_unraisable("onnxruntime.InferenceSession.run_async callback");
  }
  run->inputs.reset();
  run->callback.reset();
  run->run_options.reset();
  // Destroying the session on a thread of its own thread pool would deadlock; the main thread releases it.
  Py_AddPendingCall(
      [](void* session) {
        Py_DECREF(static_cast<PyObject*>(session));
        return 0;
      },
      run->session.release().ptr());
}

void RunAsync(nb::handle self, std::vector<std::string> input_names, nb::list inputs,
              std::vector<std::string> output_names, nb::object py_run_options, nb::callable callback) {
  Session& s = nb::cast<Session&>(self);
  const Ort::RunOptions* run_options = py_run_options.is_none() ? nullptr : &nb::cast<const Ort::RunOptions&>(py_run_options);
  auto run = std::make_unique<AsyncRun>();
  run->input_names = std::move(input_names);
  run->output_names = std::move(output_names);
  run->input_name_ptrs = CStrings(run->input_names);
  run->output_name_ptrs = CStrings(run->output_names);
  run->input_values = RawValues(nb::cast<std::vector<const Value*>>(inputs));
  run->outputs.assign(run->output_names.size(), nullptr);
  run->inputs = inputs;
  run->callback = callback;
  run->session = nb::borrow(self);
  run->run_options = py_run_options;
  AsyncRun* raw = run.get();
  Ort::ThrowOnError(Ort::GetApi().RunAsync(s.session, RawRunOptions(run_options), raw->input_name_ptrs.data(),
                                           raw->input_values.data(), raw->input_values.size(),
                                           raw->output_name_ptrs.data(), raw->output_name_ptrs.size(),
                                           raw->outputs.data(), &RunAsyncDone, raw));
  run.release();
}

nb::dict ModelMetadata(const Session& s) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto metadata = s.session.GetModelMetadata();
  nb::dict custom;
  for (const auto& key : metadata.GetCustomMetadataMapKeysAllocated(allocator)) {
    custom[key.get()] = std::string(metadata.LookupCustomMetadataMapAllocated(key.get(), allocator).get());
  }
  nb::dict result;
  result["producer_name"] = std::string(metadata.GetProducerNameAllocated(allocator).get());
  result["graph_name"] = std::string(metadata.GetGraphNameAllocated(allocator).get());
  result["domain"] = std::string(metadata.GetDomainAllocated(allocator).get());
  result["description"] = std::string(metadata.GetDescriptionAllocated(allocator).get());
  result["graph_description"] = std::string(metadata.GetGraphDescriptionAllocated(allocator).get());
  result["version"] = metadata.GetVersion();
  result["custom_metadata_map"] = custom;
  return result;
}

template <typename F>
nb::list MemoryInfos(nb::handle self, size_t count, F get) {
  std::vector<const OrtMemoryInfo*> infos(count, nullptr);
  Ort::ThrowOnError(get(nb::cast<Session&>(self).session, infos.data(), count));
  nb::list result;
  for (const OrtMemoryInfo* info : infos) {
    result.append(info ? nb::cast(MemoryInfo(info, nb::borrow(self))) : nb::none());
  }
  return result;
}

// Returns a list of (EP name, [(node name, domain, op type)]).
nb::list GraphAssignment(const Session& s) {
  const OrtApi& api = Ort::GetApi();
  const OrtEpAssignedSubgraph* const* subgraphs = nullptr;
  size_t num_subgraphs = 0;
  Ort::ThrowOnError(api.Session_GetEpGraphAssignmentInfo(s.session, &subgraphs, &num_subgraphs));
  nb::list result;
  for (size_t i = 0; i < num_subgraphs; ++i) {
    const char* ep_name = nullptr;
    Ort::ThrowOnError(api.EpAssignedSubgraph_GetEpName(subgraphs[i], &ep_name));
    const OrtEpAssignedNode* const* nodes = nullptr;
    size_t num_nodes = 0;
    Ort::ThrowOnError(api.EpAssignedSubgraph_GetNodes(subgraphs[i], &nodes, &num_nodes));
    nb::list node_list;
    for (size_t j = 0; j < num_nodes; ++j) {
      const char* name = nullptr;
      const char* domain = nullptr;
      const char* op_type = nullptr;
      Ort::ThrowOnError(api.EpAssignedNode_GetName(nodes[j], &name));
      Ort::ThrowOnError(api.EpAssignedNode_GetDomain(nodes[j], &domain));
      Ort::ThrowOnError(api.EpAssignedNode_GetOperatorType(nodes[j], &op_type));
      node_list.append(nb::make_tuple(name, domain, op_type));
    }
    result.append(nb::make_tuple(ep_name, node_list));
  }
  return result;
}

// Returns `options` with a copy of `state`, to which the callbacks of `options` are redirected.
SessionOptions WithOwnState(Ort::SessionOptions options, const SessionOptionsState& state) {
  SessionOptions copy{std::move(options), std::make_shared<SessionOptionsState>(state)};
  if (copy.state->ep_context_data_read_func.is_valid()) {
    Ort::ThrowOnError(
        Ort::GetApi().SessionOptionsSetEpContextDataReadFunc(copy.options, &ReadEpContextData, copy.state.get()));
  }
  if (copy.state->selection_delegate.is_valid()) {
    Ort::ThrowOnError(
        Ort::GetApi().SessionOptionsSetEpSelectionPolicyDelegate(copy.options, &SelectEpDevices, copy.state.get()));
  }
  return copy;
}

// Returns a copy of `o` whose callbacks are independent of `o`.
SessionOptions CloneOptions(const SessionOptions& o) { return WithOwnState(o.options.Clone(), *o.state); }

struct ModelCompiler {
  Ort::ModelCompilationOptions options{nullptr};
  // The session options that `options` refers to.
  SessionOptions session_options;
  nb::object input_model;
  nb::object get_initializer_location_func;
  nb::object ep_context_data_write_func;
  nb::object stream_write_func;
  bool compiling = false;
};

OrtStatus* ORT_API_CALL GetInitializerLocation(void* state, const char* name, const OrtValue* value,
                                               const OrtExternalInitializerInfo* external_info,
                                               OrtExternalInitializerInfo** new_external_info) {
  *new_external_info = nullptr;
  return CallPython([&]() -> OrtStatus* {
    auto* compiler = static_cast<ModelCompiler*>(state);
    nb::object info = nb::none();
    if (external_info != nullptr) {
      Ort::ConstExternalInitializerInfo i{external_info};
      info = nb::make_tuple(std::string(i.GetFilePath()), i.GetFileOffset(), i.GetByteSize());
    }
    // The initializer is only valid during the call; `value` views it without owning it.
    Value view{{}, Ort::Value(const_cast<OrtValue*>(value))};
    nb::object py_value = nb::cast(&view, nb::rv_policy::reference);
    nb::object result;
    try {
      result = compiler->get_initializer_location_func(name, py_value, info);
    } catch (...) {
      view.value.release();
      throw;
    }
    view.value.release();
    if (!result.is_none()) {
      auto [path, offset, size] = nb::cast<std::tuple<std::string, int64_t, size_t>>(result);
      Ort::ThrowOnError(Ort::GetApi().CreateExternalInitializerInfo(
          std::filesystem::path(path).c_str(), offset, size, new_external_info));
    }
    return nullptr;
  });
}

OrtStatus* ORT_API_CALL WriteEpContextData(void* state, const char* name, const void* buffer, size_t size) {
  return CallPython([&]() -> OrtStatus* {
    auto* compiler = static_cast<ModelCompiler*>(state);
    auto data = std::make_shared<EpContextData>(EpContextData{buffer, size});
    try {
      compiler->ep_context_data_write_func(name, data);
    } catch (...) {
      data->valid = false;
      throw;
    }
    data->valid = false;
    return nullptr;
  });
}

OrtStatus* ORT_API_CALL WriteModelToStream(void* state, const void* buffer, size_t size) {
  return CallPython([&]() -> OrtStatus* {
    static_cast<ModelCompiler*>(state)->stream_write_func(nb::bytes(buffer, size));
    return nullptr;
  });
}

// Throws if `compiler` is compiling; its output settings must not change during a compilation.
void CheckNotCompiling(const ModelCompiler& compiler) {
  if (compiler.compiling) {
    throw Ort::Exception("Compilation is already in progress", ORT_FAIL);
  }
}

void Compile(ModelCompiler& compiler) {
  OrtStatus* status = nullptr;
  compiler.compiling = true;
  {
    nb::gil_scoped_release release;
    status = Ort::GetCompileApi().CompileModel(GetEnv(), compiler.options);
  }
  compiler.compiling = false;
  Ort::ThrowOnError(status);
}

}  // namespace

void BindSession(nb::module_& m) {
  using Options = SessionOptions;
  nb::class_<Options>(m, "SessionOptions")
      .def(nb::init<>())
      .def(
          "clone",
          &CloneOptions,
          "Return a copy whose callbacks are independent of these options.")
      .def_prop_ro("intra_op_num_threads", [](const Options& o) { return o.options.GetIntraOpNumThreads(); })
      .def_prop_ro("inter_op_num_threads", [](const Options& o) { return o.options.GetInterOpNumThreads(); })
      .def_prop_ro("graph_optimization_level",
                   [](const Options& o) { return static_cast<int>(o.options.GetGraphOptimizationLevel()); })
      .def_prop_ro("execution_mode", [](const Options& o) { return static_cast<int>(o.options.GetExecutionMode()); })
      .def_prop_ro("enable_profiling", [](const Options& o) { return o.options.GetProfilingEnabled(); })
      .def_prop_ro("profile_file_prefix", [](const Options& o) { return o.options.GetProfileFilePrefix(); })
      .def_prop_ro("optimized_model_filepath", [](const Options& o) { return o.options.GetOptimizedModelFilePath(); })
      .def_prop_ro("enable_cpu_mem_arena", [](const Options& o) { return o.options.GetCpuMemArenaEnabled(); })
      .def_prop_ro("enable_mem_pattern", [](const Options& o) { return o.options.GetMemPatternEnabled(); })
      .def_prop_ro("logid", [](const Options& o) { return o.options.GetLogId(); })
      .def_prop_ro("log_severity_level", [](const Options& o) { return o.options.GetLogSeverityLevel(); })
      .def_prop_ro("log_verbosity_level", [](const Options& o) { return o.options.GetLogVerbosityLevel(); })
      .def_prop_ro("use_per_session_threads", [](const Options& o) { return o.options.GetPerSessionThreadsEnabled(); })
      .def_prop_ro("use_deterministic_compute",
                   [](const Options& o) { return o.options.GetDeterministicComputeEnabled(); })
      .def_prop_ro("config_entries", [](const Options& o) { return o.options.GetConfigEntries().GetKeyValuePairs(); })
      .def("set_intra_op_num_threads", [](Options& o, int n) { o.options.SetIntraOpNumThreads(n); })
      .def("set_inter_op_num_threads", [](Options& o, int n) { o.options.SetInterOpNumThreads(n); })
      .def("set_graph_optimization_level",
           [](Options& o, int level) {
             o.options.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(level));
           })
      .def("set_execution_mode",
           [](Options& o, int mode) { o.options.SetExecutionMode(static_cast<ExecutionMode>(mode)); })
      .def("set_log_severity_level", [](Options& o, int level) { o.options.SetLogSeverityLevel(level); })
      .def("set_log_verbosity_level",
           [](Options& o, int level) {
             Ort::ThrowOnError(Ort::GetApi().SetSessionLogVerbosityLevel(o.options, level));
           })
      .def("set_logid", [](Options& o, const std::string& logid) { o.options.SetLogId(logid.c_str()); })
      .def("set_profiling",
           [](Options& o, bool enable, const std::filesystem::path& prefix) {
             if (enable) {
               o.options.EnableProfiling(prefix.c_str());
             } else {
               o.options.DisableProfiling();
             }
           })
      .def("set_optimized_model_filepath",
           [](Options& o, const std::filesystem::path& path) { o.options.SetOptimizedModelFilePath(path.c_str()); })
      .def("set_cpu_mem_arena",
           [](Options& o, bool enable) { enable ? o.options.EnableCpuMemArena() : o.options.DisableCpuMemArena(); })
      .def("set_mem_pattern",
           [](Options& o, bool enable) { enable ? o.options.EnableMemPattern() : o.options.DisableMemPattern(); })
      .def("disable_per_session_threads", [](Options& o) { o.options.DisablePerSessionThreads(); })
      .def("set_deterministic_compute", [](Options& o, bool value) { o.options.SetDeterministicCompute(value); })
      .def("set_load_cancellation_flag", [](Options& o, bool value) { o.options.SetLoadCancellationFlag(value); })
      .def("add_config_entry",
           [](Options& o, const std::string& key, const std::string& value) {
             o.options.AddConfigEntry(key.c_str(), value.c_str());
           })
      .def("add_free_dimension_override_by_name",
           [](Options& o, const std::string& name, int64_t value) {
             o.options.AddFreeDimensionOverrideByName(name.c_str(), value);
           })
      .def("add_free_dimension_override_by_denotation",
           [](Options& o, const std::string& denotation, int64_t value) {
             o.options.AddFreeDimensionOverride(denotation.c_str(), value);
           })
      .def("register_custom_ops_library",
           [](Options& o, const std::filesystem::path& path) { o.options.RegisterCustomOpsLibrary(path.c_str()); })
      .def("add_initializer",
           [](Options& o, const std::string& name, nb::handle value) {
             o.options.AddInitializer(name.c_str(), nb::cast<const Value&>(value).value);
             o.state->keep_alive.append(value);
           })
      .def("add_external_initializers",
           [](Options& o, const std::vector<std::string>& names, const std::vector<nb::handle>& values) {
             auto name_ptrs = CStrings(names);
             std::vector<const OrtValue*> value_ptrs;
             for (nb::handle value : values) {
               value_ptrs.push_back(nb::cast<const Value&>(value).value);
               o.state->keep_alive.append(value);
             }
             Ort::ThrowOnError(Ort::GetApi().AddExternalInitializers(o.options, name_ptrs.data(), value_ptrs.data(),
                                                                     value_ptrs.size()));
           })
      .def("add_external_initializers_from_files_in_memory",
           [](Options& o, const std::vector<std::string>& names, const std::vector<nb::bytes>& buffers) {
             std::vector<std::basic_string<ORTCHAR_T>> file_names;
             std::vector<char*> data;
             std::vector<size_t> lengths;
             for (size_t i = 0; i < names.size(); ++i) {
               file_names.push_back(std::filesystem::path(names[i]).native());
               data.push_back(const_cast<char*>(buffers[i].c_str()));
               lengths.push_back(buffers[i].size());
               o.state->keep_alive.append(buffers[i]);
             }
             o.options.AddExternalInitializersFromFilesInMemory(file_names, data, lengths);
           })
      .def("append_execution_provider",
           [](Options& o, const std::string& name, const std::map<std::string, std::string>& options) {
             if (name == "CUDAExecutionProvider") {
               Ort::CUDAProviderOptions cuda;
               cuda.Update(StringMap(options.begin(), options.end()));
               o.options.AppendExecutionProvider_CUDA_V2(*cuda);
             } else if (name == "TensorrtExecutionProvider") {
               Ort::TensorRTProviderOptions trt;
               trt.Update(StringMap(options.begin(), options.end()));
               o.options.AppendExecutionProvider_TensorRT_V2(*trt);
             } else {
               o.options.AppendExecutionProvider(name, StringMap(options.begin(), options.end()));
             }
           })
      .def("append_execution_provider_for_devices",
           [](Options& o, const std::vector<EpDevice>& devices, const std::map<std::string, std::string>& options) {
             std::vector<const OrtEpDevice*> raw;
             for (const auto& d : devices) {
               raw.push_back(d.ptr);
             }
             std::vector<const char*> keys, values;
             for (const auto& [k, v] : options) {
               keys.push_back(k.c_str());
               values.push_back(v.c_str());
             }
             Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
                 o.options, GetEnv(), raw.data(), raw.size(), keys.data(), values.data(), keys.size()));
           })
      .def("set_ep_selection_policy",
           [](Options& o, int policy) {
             o.state->selection_delegate.reset();
             Ort::ThrowOnError(Ort::GetApi().SessionOptionsSetEpSelectionPolicy(
                 o.options, static_cast<OrtExecutionProviderDevicePolicy>(policy)));
           })
      .def("set_ep_selection_policy_delegate",
           [](Options& o, nb::callable delegate) {
             o.state->selection_delegate = delegate;
             Ort::ThrowOnError(
                 Ort::GetApi().SessionOptionsSetEpSelectionPolicyDelegate(o.options, &SelectEpDevices, o.state.get()));
           })
      .def(
          "set_ep_context_data_read_func",
          [](Options& o, nb::object read_func, size_t max_data_size) {
            if (read_func.is_none()) {
              o.state->ep_context_data_read_func.reset();
              Ort::ThrowOnError(Ort::GetApi().SessionOptionsSetEpContextDataReadFunc(o.options, nullptr, nullptr));
              return;
            }
            if (max_data_size == 0 || max_data_size == std::numeric_limits<size_t>::max()) {
              throw Ort::Exception("EPContext data max_data_size must be finite and greater than zero.",
                                   ORT_INVALID_ARGUMENT);
            }
            o.state->ep_context_data_read_func = read_func;
            o.state->ep_context_data_max_size = max_data_size;
            Ort::ThrowOnError(
                Ort::GetApi().SessionOptionsSetEpContextDataReadFunc(o.options, &ReadEpContextData, o.state.get()));
          },
          nb::arg("read_func").none(), nb::arg("max_data_size") = 0)
      .def("get_config_entry",
           [](const Options& o, const std::string& key) -> std::optional<std::string> {
             if (!o.options.HasConfigEntry(key.c_str())) {
               return std::nullopt;
             }
             return o.options.GetConfigEntry(key.c_str());
           });

  nb::class_<EpContextDataBuffer>(m, "OrtEpContextDataBuffer")
      .def_prop_ro("size",
                   [](const EpContextDataBuffer& b) {
                     b.CheckValid();
                     return b.size;
                   })
      .def_prop_ro("max_size",
                   [](const EpContextDataBuffer& b) {
                     b.CheckValid();
                     return b.max_size;
                   })
      .def_prop_ro("is_allocated",
                   [](const EpContextDataBuffer& b) {
                     b.CheckValid();
                     return b.allocated;
                   })
      .def(
          "allocate",
          [](EpContextDataBuffer& b, size_t size) {
            b.CheckValid();
            if (b.allocated) {
              throw std::runtime_error("EPContext data output buffer is already allocated.");
            }
            if (size > b.max_size) {
              throw std::runtime_error("EPContext data size " + std::to_string(size) + " exceeds the configured maximum size " +
                                       std::to_string(b.max_size) + ".");
            }
            b.data = size > 0 ? b.allocator->Alloc(b.allocator, size) : nullptr;
            b.size = size;
            b.allocated = true;
          },
          nb::arg("size"), "Allocate the final ORT-owned output buffer.")
      .def(
          "write",
          [](EpContextDataBuffer& b, nb::bytes data, size_t offset) {
            b.CheckValid();
            if (!b.allocated) {
              throw std::runtime_error("EPContext data output buffer must be allocated before writing.");
            }
            if (offset > b.size || data.size() > b.size - offset) {
              throw std::runtime_error("EPContext data write is out of bounds.");
            }
            if (data.size() > 0) {
              std::memcpy(static_cast<char*>(b.data) + offset, data.c_str(), data.size());
            }
          },
          nb::arg("data"), nb::arg("offset") = 0, "Copy a chunk into the allocated output.");

  nb::class_<EpContextData>(m, "OrtEpContextData")
      .def_prop_ro("size",
                   [](const EpContextData& d) {
                     if (!d.valid) {
                       throw std::runtime_error("the EPContext data is only valid during its write callback");
                     }
                     return d.size;
                   })
      .def(
          "read",
          [](const EpContextData& d, size_t offset, std::optional<size_t> length) {
            if (!d.valid) {
              throw std::runtime_error("the EPContext data is only valid during its write callback");
            }
            if (offset > d.size) {
              throw std::runtime_error("EPContext data read offset is out of bounds.");
            }
            size_t n = std::min(length.value_or(d.size - offset), d.size - offset);
            return nb::bytes(static_cast<const char*>(d.data) + offset, n);
          },
          nb::arg("offset") = 0, nb::arg("length").none() = nb::none(), "Copy a range of the data into bytes.");

  using RunOptions = Ort::RunOptions;
  nb::class_<RunOptions>(m, "RunOptions")
      .def(nb::init<>())
      .def_prop_rw(
          "log_severity_level", [](const RunOptions& o) { return o.GetRunLogSeverityLevel(); },
          [](RunOptions& o, int level) { o.SetRunLogSeverityLevel(level); })
      .def_prop_rw(
          "log_verbosity_level", [](const RunOptions& o) { return o.GetRunLogVerbosityLevel(); },
          [](RunOptions& o, int level) { o.SetRunLogVerbosityLevel(level); })
      .def_prop_rw(
          "logid", [](const RunOptions& o) { return std::string(o.GetRunTag()); },
          [](RunOptions& o, const std::string& tag) { o.SetRunTag(tag.c_str()); })
      .def("set_terminate", [](RunOptions& o, bool terminate) { terminate ? o.SetTerminate() : o.UnsetTerminate(); })
      .def("add_config_entry",
           [](RunOptions& o, const std::string& key, const std::string& value) {
             o.AddConfigEntry(key.c_str(), value.c_str());
           })
      .def("get_config_entry",
           [](RunOptions& o, const std::string& key) -> std::optional<std::string> {
             const char* value = o.GetConfigEntry(key.c_str());
             if (value == nullptr) {
               return std::nullopt;
             }
             return value;
           })
      .def("add_active_lora_adapter",
           [](RunOptions& o, const Ort::LoraAdapter& adapter) { o.AddActiveLoraAdapter(adapter); });

  nb::class_<Session>(m, "Session")
      .def(
          "__init__",
          [](Session* self, const std::filesystem::path& model_path, const Options& options) {
            Ort::Env& env = GetEnv();
            nb::gil_scoped_release release;
            new (self) Session{Ort::Session(env, model_path.c_str(), options.options), options.state};
          },
          nb::arg("model_path"), nb::arg("options"))
      .def(
          "__init__",
          [](Session* self, nb::bytes model, const Options& options) {
            Ort::Env& env = GetEnv();
            nb::gil_scoped_release release;
            new (self) Session{Ort::Session(env, model.data(), model.size(), options.options), options.state};
          },
          nb::arg("model"), nb::arg("options"))
      .def(
          "session_options",
          [](const Session& s) { return WithOwnState(s.session.GetSessionOptions(), *s.options_state); },
          "Return a copy of the options the session uses, without execution providers.")
      .def_prop_ro("inputs",
                   [](const Session& s) {
                     return DescribeArgs(
                         s.session.GetInputCount(),
                         [&](size_t i, auto& a) { return s.session.GetInputNameAllocated(i, a); },
                         [&](size_t i) { return s.session.GetInputTypeInfo(i); });
                   })
      .def_prop_ro("outputs",
                   [](const Session& s) {
                     return DescribeArgs(
                         s.session.GetOutputCount(),
                         [&](size_t i, auto& a) { return s.session.GetOutputNameAllocated(i, a); },
                         [&](size_t i) { return s.session.GetOutputTypeInfo(i); });
                   })
      .def_prop_ro("overridable_initializers",
                   [](const Session& s) {
                     return DescribeArgs(
                         s.session.GetOverridableInitializerCount(),
                         [&](size_t i, auto& a) { return s.session.GetOverridableInitializerNameAllocated(i, a); },
                         [&](size_t i) { return s.session.GetOverridableInitializerTypeInfo(i); });
                   })
      .def_prop_ro("metadata", &ModelMetadata)
      .def_prop_ro("input_memory_infos",
                   [](nb::handle self) {
                     return MemoryInfos(self, nb::cast<Session&>(self).session.GetInputCount(),
                                        [](OrtSession* s, const OrtMemoryInfo** out, size_t n) {
                                          return Ort::GetApi().SessionGetMemoryInfoForInputs(s, out, n);
                                        });
                   })
      .def_prop_ro("output_memory_infos",
                   [](nb::handle self) {
                     return MemoryInfos(self, nb::cast<Session&>(self).session.GetOutputCount(),
                                        [](OrtSession* s, const OrtMemoryInfo** out, size_t n) {
                                          return Ort::GetApi().SessionGetMemoryInfoForOutputs(s, out, n);
                                        });
                   })
      .def_prop_ro("input_ep_devices",
                   [](const Session& s) {
                     size_t count = s.session.GetInputCount();
                     std::vector<const OrtEpDevice*> devices(count, nullptr);
                     Ort::ThrowOnError(Ort::GetApi().SessionGetEpDeviceForInputs(s.session, devices.data(), count));
                     nb::list result;
                     for (const OrtEpDevice* d : devices) {
                       result.append(d ? nb::cast(EpDevice{d}) : nb::none());
                     }
                     return result;
                   })
      .def_prop_ro("provider_names", [](const Session& s) { return s.session.GetProviderNames(); })
      .def(
          "provider_options",
          [](const Session& s, size_t index) { return s.session.GetProviderOptions(index).GetKeyValuePairs(); },
          nb::arg("index"))
      .def("tuning_results_json",
           [](const Session& s) {
             Ort::AllocatorWithDefaultOptions allocator;
             return std::string(s.session.GetTuningResultsAllocated(allocator).get());
           })
      .def(
          "set_tuning_results_json",
          [](Session& s, const std::string& json, bool error_on_invalid) {
            s.session.SetTuningResults(json.c_str(), error_on_invalid);
          },
          nb::arg("json"), nb::arg("error_on_invalid"))
      .def("end_profiling",
           [](Session& s) {
             Ort::AllocatorWithDefaultOptions allocator;
             return std::string(s.session.EndProfilingAllocated(allocator).get());
           })
      .def_prop_ro("profiling_start_time_ns", [](const Session& s) { return s.session.GetProfilingStartTimeNs(); })
      .def("set_ep_dynamic_options",
           [](Session& s, const std::map<std::string, std::string>& options) {
             std::vector<const char*> keys, values;
             for (const auto& [k, v] : options) {
               keys.push_back(k.c_str());
               values.push_back(v.c_str());
             }
             s.session.SetEpDynamicOptions(keys.data(), values.data(), keys.size());
           })
      .def("graph_assignment", &GraphAssignment)
      .def("release_captured_graph",
           [](Session& s, int graph_annotation_id) {
             Ort::ThrowOnError(Ort::GetApi().SessionReleaseCapturedGraph(s.session, graph_annotation_id));
           })
      .def(
          "allocate",
          [](nb::handle self, const std::vector<int64_t>& shape, int element_type, const MemoryInfo& memory_info) {
            auto& s = nb::cast<Session&>(self);
            auto allocator = std::make_unique<Ort::Allocator>(s.session, memory_info.ptr);
            auto value = Ort::Value::CreateTensor(*allocator, shape.data(), shape.size(),
                                                  static_cast<ONNXTensorElementDataType>(element_type));
            // The tensor frees its buffer with the allocator, which belongs to the session. Tuples release their items
            // in reverse order, so the allocator is released before the session.
            nb::capsule allocator_owner(allocator.release(),
                                        [](void* p) noexcept { delete static_cast<Ort::Allocator*>(p); });
            return Value{nb::make_tuple(self, allocator_owner), std::move(value)};
          },
          nb::arg("shape"), nb::arg("element_type"), nb::arg("memory_info"),
          "Return an uninitialized tensor allocated by the allocator of this session for `memory_info`.")
      .def("run", &Run, nb::arg("input_names"), nb::arg("inputs"), nb::arg("output_names"),
           nb::arg("run_options").none())
      .def("run_async", &RunAsync, nb::arg("input_names"), nb::arg("inputs"), nb::arg("output_names"),
           nb::arg("run_options").none(), nb::arg("callback"),
           "Run the session on a thread of ONNX Runtime and call `callback(outputs, error)` when done.")
      .def(
          "run_with_binding",
          [](Session& s, const Ort::IoBinding& binding, const Ort::RunOptions* run_options) {
            OrtStatus* status = nullptr;
            {
              nb::gil_scoped_release release;
              status = Ort::GetApi().RunWithBinding(s.session, RawRunOptions(run_options), binding);
            }
            Ort::ThrowOnError(status);
          },
          nb::arg("binding"), nb::arg("run_options").none());

  nb::class_<Ort::IoBinding>(m, "IOBinding")
      .def(
          "__init__", [](Ort::IoBinding* self, Session& s) { new (self) Ort::IoBinding(s.session); }, nb::arg("session"),
          nb::keep_alive<1, 2>())
      .def(
          "bind_input",
          [](Ort::IoBinding& b, const std::string& name, const Value& value) { b.BindInput(name.c_str(), value.value); },
          nb::arg("name"), nb::arg("value"))
      .def(
          "bind_output",
          [](Ort::IoBinding& b, const std::string& name, const Value& value) {
            b.BindOutput(name.c_str(), value.value);
          },
          nb::arg("name"), nb::arg("value"))
      .def(
          "bind_output_to_device",
          [](Ort::IoBinding& b, const std::string& name, const MemoryInfo& memory_info) {
            b.BindOutput(name.c_str(), memory_info.ptr);
          },
          nb::arg("name"), nb::arg("memory_info"))
      .def("output_names", [](const Ort::IoBinding& b) { return b.GetOutputNames(); })
      .def("outputs",
           [](const Ort::IoBinding& b) {
             std::vector<Value> result;
             for (auto& value : b.GetOutputValues()) {
               result.push_back({{}, std::move(value)});
             }
             return result;
           })
      .def("clear_inputs", [](Ort::IoBinding& b) { b.ClearBoundInputs(); })
      .def("clear_outputs", [](Ort::IoBinding& b) { b.ClearBoundOutputs(); })
      .def("synchronize_inputs", [](Ort::IoBinding& b) { b.SynchronizeInputs(); })
      .def("synchronize_outputs", [](Ort::IoBinding& b) { b.SynchronizeOutputs(); });

  nb::class_<ModelCompiler>(m, "ModelCompiler")
      .def(
          "__init__",
          [](ModelCompiler* self, const Options& options, nb::object model, bool embed_compiled_data_into_model,
             const std::filesystem::path& external_initializers_file_path, size_t external_initializers_size_threshold,
             uint32_t flags, int graph_optimization_level, nb::object get_initializer_location_func) {
            new (self) ModelCompiler{Ort::ModelCompilationOptions{nullptr}, CloneOptions(options)};
            self->options = Ort::ModelCompilationOptions(GetEnv(), self->session_options.options);
            auto& o = self->options;
            if (nb::isinstance<nb::bytes>(model)) {
              auto bytes = nb::cast<nb::bytes>(model);
              o.SetInputModelFromBuffer(bytes.data(), bytes.size());
            } else {
              o.SetInputModelPath(nb::cast<std::filesystem::path>(model).c_str());
            }
            self->input_model = model;
            o.SetEpContextEmbedMode(embed_compiled_data_into_model);
            if (!external_initializers_file_path.empty()) {
              o.SetOutputModelExternalInitializersFile(external_initializers_file_path.c_str(),
                                                       external_initializers_size_threshold);
            }
            o.SetFlags(flags);
            o.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(graph_optimization_level));
            if (!get_initializer_location_func.is_none()) {
              self->get_initializer_location_func = get_initializer_location_func;
              o.SetOutputModelGetInitializerLocationFunc(&GetInitializerLocation, self);
            }
          },
          nb::arg("options"), nb::arg("model"), nb::arg("embed_compiled_data_into_model"),
          nb::arg("external_initializers_file_path"), nb::arg("external_initializers_size_threshold"),
          nb::arg("flags"), nb::arg("graph_optimization_level"), nb::arg("get_initializer_location_func").none())
      .def(
          "set_ep_context_data_write_func",
          [](ModelCompiler& c, nb::object write_func) {
            if (c.compiling) {
              throw std::runtime_error("the EPContext data write callback cannot change while compilation is in progress");
            }
            c.ep_context_data_write_func = write_func;
            c.options.SetEpContextDataWriteFunc(write_func.is_none() ? nullptr : &WriteEpContextData,
                                                write_func.is_none() ? nullptr : &c);
          },
          nb::arg("write_func").none())
      .def("set_ep_context_binary_information",
           [](ModelCompiler& c, const std::filesystem::path& output_directory, const std::filesystem::path& model_name) {
             c.options.SetEpContextBinaryInformation(output_directory.c_str(), model_name.c_str());
           })
      .def(
          "compile_to_file",
          [](ModelCompiler& c, std::optional<std::filesystem::path> path) {
            CheckNotCompiling(c);
            if (path) {
              c.options.SetOutputModelPath(path->c_str());
            }
            Compile(c);
          },
          nb::arg("path").none())
      .def("compile_to_bytes",
           [](ModelCompiler& c) {
             CheckNotCompiling(c);
             Ort::AllocatorWithDefaultOptions allocator;
             void* buffer = nullptr;
             size_t size = 0;
             c.options.SetOutputModelBuffer(allocator, &buffer, &size);
             Compile(c);
             nb::bytes result(buffer, size);
             allocator.Free(buffer);
             return result;
           })
      .def("compile_to_stream", [](ModelCompiler& c, nb::callable write_func) {
        CheckNotCompiling(c);
        c.stream_write_func = write_func;
        c.options.SetOutputModelWriteFunc(&WriteModelToStream, &c);
        Compile(c);
      });
}

}  // namespace onnxruntime_nb
