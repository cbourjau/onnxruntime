// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Native part of the onnxruntime_nb package: bindings of the public C++ API.
// Conveniences, defaults, and exception classes live in the Python package.

#include <exception>
#include <filesystem>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "onnxruntime_cxx_api.h"
#include "value.h"

namespace nb = nanobind;
using namespace onnxruntime_nb;

namespace {

// Raises the Python exception class that `onnxruntime_nb._exceptions` assigns to the error code.
void RegisterExceptionTranslator() {
  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const Ort::Exception& e) {
      nb::object type = nb::module_::import_("onnxruntime_nb._exceptions")
                            .attr("exception_class")(static_cast<int>(e.GetOrtErrorCode()));
      PyErr_SetString(type.ptr(), e.what());
    }
  });
}

// Runs `session` with the GIL released.
std::vector<Ort::Value> Run(Ort::Session& session, const std::vector<std::string>& input_names,
                            const std::vector<const Ort::Value*>& inputs, const std::vector<std::string>& output_names) {
  std::vector<const char*> input_name_ptrs;
  for (const auto& name : input_names) {
    input_name_ptrs.push_back(name.c_str());
  }
  // Ort::Session::Run takes a contiguous array of Ort::Value, which would own the inputs; hence the C function.
  std::vector<const OrtValue*> input_values;
  for (const Ort::Value* value : inputs) {
    input_values.push_back(*value);
  }
  std::vector<const char*> output_name_ptrs;
  for (const auto& name : output_names) {
    output_name_ptrs.push_back(name.c_str());
  }
  std::vector<OrtValue*> outputs(output_names.size(), nullptr);
  OrtStatus* status = nullptr;
  {
    nb::gil_scoped_release release;
    status = Ort::GetApi().Run(session, nullptr, input_name_ptrs.data(), input_values.data(), input_values.size(),
                               output_name_ptrs.data(), output_name_ptrs.size(), outputs.data());
  }
  Ort::ThrowOnError(status);
  return {outputs.begin(), outputs.end()};
}

}  // namespace

NB_MODULE(_C, m) {
  RegisterExceptionTranslator();

  m.def("get_available_providers", &Ort::GetAvailableProviders,
        "Return the names of the execution providers available in this build.");
  m.def("set_seed", &Ort::SetGlobalRandomSeed, nb::arg("seed"),
        "Set the process-wide seed of random operators that have no `seed` attribute; "
        "call it before creating sessions to make their outputs reproducible.");

  m.def("get_device", &Ort::GetBuildDeviceString,
        "Return a string describing the devices this build targets, e.g. 'CPU' or 'GPU-DNNL'.");
  m.def("get_all_providers", &Ort::GetAllProviders,
        "Return the names of all execution providers that ONNX Runtime supports, in their default priority order.");

  nb::class_<Ort::Env>(m, "Env")
      .def(nb::init<>())
      .def("update_log_severity_level",
           [](Ort::Env& e, int level) { e.UpdateEnvWithCustomLogLevel(static_cast<OrtLoggingLevel>(level)); })
      .def("update_log_verbosity_level", [](Ort::Env& e, int level) { e.UpdateEnvWithCustomLogVerbosityLevel(level); });

  using Options = Ort::SessionOptions;
  nb::class_<Options>(m, "SessionOptions")
      .def(nb::init<>())
      .def("set_intra_op_num_threads", [](Options& o, int n) { o.SetIntraOpNumThreads(n); })
      .def("set_inter_op_num_threads", [](Options& o, int n) { o.SetInterOpNumThreads(n); })
      .def("set_graph_optimization_level",
           [](Options& o, int level) { o.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(level)); })
      .def("set_execution_mode", [](Options& o, int mode) { o.SetExecutionMode(static_cast<ExecutionMode>(mode)); })
      .def("set_log_severity_level", [](Options& o, int level) { o.SetLogSeverityLevel(level); })
      // The C++ API has no wrapper for SetSessionLogVerbosityLevel.
      .def("set_log_verbosity_level",
           [](Options& o, int level) { Ort::ThrowOnError(Ort::GetApi().SetSessionLogVerbosityLevel(o, level)); })
      .def("set_logid", [](Options& o, const std::string& logid) { o.SetLogId(logid.c_str()); })
      .def("enable_profiling",
           [](Options& o, const std::filesystem::path& prefix) { o.EnableProfiling(prefix.c_str()); })
      .def("set_optimized_model_filepath",
           [](Options& o, const std::filesystem::path& path) { o.SetOptimizedModelFilePath(path.c_str()); })
      .def("set_cpu_mem_arena",
           [](Options& o, bool enable) { enable ? o.EnableCpuMemArena() : o.DisableCpuMemArena(); })
      .def("set_mem_pattern", [](Options& o, bool enable) { enable ? o.EnableMemPattern() : o.DisableMemPattern(); })
      .def("add_config_entry",
           [](Options& o, const std::string& key, const std::string& value) {
             o.AddConfigEntry(key.c_str(), value.c_str());
           })
      .def("append_execution_provider",
           [](Options& o, const std::string& name, const std::unordered_map<std::string, std::string>& options) {
             o.AppendExecutionProvider(name, options);
           });

  nb::class_<Ort::Value>(m, "OrtValue")
      .def_static("from_array", &TensorFromArray, nb::arg("array"), nb::keep_alive<0, 1>(),
                  "Return a tensor that borrows the buffer of `array`.")
      .def_static("from_strings", &TensorFromStrings, nb::arg("strings"), nb::arg("shape"))
      .def_prop_ro("element_type",
                   [](const Ort::Value& v) { return static_cast<int>(v.GetTensorTypeAndShapeInfo().GetElementType()); })
      .def_prop_ro("shape", [](const Ort::Value& v) { return v.GetTensorTypeAndShapeInfo().GetShape(); })
      .def(
          "numpy", [](nb::handle self) { return TensorToArray(nb::cast<Ort::Value&>(self), self); },
          "Return a numpy array that views the tensor's buffer and keeps this value alive.")
      .def("strings", &TensorToStrings);

  nb::class_<Ort::Session>(m, "Session")
      .def(
          "__init__",
          [](Ort::Session* self, const Ort::Env& env, const std::filesystem::path& model_path, const Options& options) {
            new (self) Ort::Session(env, model_path.c_str(), options);
          },
          nb::arg("env"), nb::arg("model_path"), nb::arg("options"), nb::keep_alive<1, 2>())
      .def_prop_ro("input_names", [](const Ort::Session& s) { return s.GetInputNames(); })
      .def_prop_ro("output_names", [](const Ort::Session& s) { return s.GetOutputNames(); })
      .def_prop_ro("provider_names", [](const Ort::Session& s) { return s.GetProviderNames(); })
      .def(
          "provider_options",
          [](const Ort::Session& s, size_t index) { return s.GetProviderOptions(index).GetKeyValuePairs(); },
          nb::arg("index"))
      .def("tuning_results_json",
           [](const Ort::Session& s) {
             Ort::AllocatorWithDefaultOptions allocator;
             return std::string(s.GetTuningResultsAllocated(allocator).get());
           })
      .def(
          "set_tuning_results_json",
          [](Ort::Session& s, const std::string& json, bool error_on_invalid) {
            s.SetTuningResults(json.c_str(), error_on_invalid);
          },
          nb::arg("json"), nb::arg("error_on_invalid"))
      .def("run", &Run, nb::arg("input_names"), nb::arg("inputs"), nb::arg("output_names"));
}
