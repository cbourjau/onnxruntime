// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Native part of the onnxruntime_nb package: a thin layer over the public C API.
// Conveniences, defaults, and exception classes live in the Python package.

#include <exception>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ort_api.h"
#include "session.h"

namespace nb = nanobind;
using namespace onnxruntime_nb;

namespace {

// Raises the Python exception class that `onnxruntime_nb._exceptions` assigns to the error code.
void RegisterExceptionTranslator() {
  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const OrtError& e) {
      nb::object type = nb::module_::import_("onnxruntime_nb._exceptions")
                            .attr("exception_class")(static_cast<int>(e.code()));
      PyErr_SetString(type.ptr(), e.what());
    }
  });
}

}  // namespace

NB_MODULE(_C, m) {
  InitApi();
  RegisterExceptionTranslator();

  m.def("get_available_providers", &GetAvailableProviders,
        "Return the names of the execution providers available in this build.");

  nb::class_<SessionOptions>(m, "SessionOptions")
      .def(nb::init<>())
      .def("set_intra_op_num_threads", &SessionOptions::SetIntraOpNumThreads)
      .def("set_inter_op_num_threads", &SessionOptions::SetInterOpNumThreads)
      .def("set_graph_optimization_level", &SessionOptions::SetGraphOptimizationLevel)
      .def("set_execution_mode", &SessionOptions::SetExecutionMode)
      .def("set_log_severity_level", &SessionOptions::SetLogSeverityLevel)
      .def("set_log_verbosity_level", &SessionOptions::SetLogVerbosityLevel)
      .def("set_logid", &SessionOptions::SetLogId)
      .def("enable_profiling", &SessionOptions::EnableProfiling)
      .def("set_optimized_model_filepath", &SessionOptions::SetOptimizedModelFilePath)
      .def("set_cpu_mem_arena", &SessionOptions::SetCpuMemArena)
      .def("set_mem_pattern", &SessionOptions::SetMemPattern)
      .def("add_config_entry", &SessionOptions::AddConfigEntry)
      .def("append_execution_provider", &SessionOptions::AppendExecutionProvider);

  nb::class_<Session>(m, "Session")
      .def(nb::init<const std::filesystem::path&, const SessionOptions&>())
      .def_prop_ro("input_names", &Session::InputNames)
      .def_prop_ro("output_names", &Session::OutputNames)
      .def("run", &Session::Run, nb::arg("output_names"), nb::arg("arrays"), nb::arg("strings"));
}
