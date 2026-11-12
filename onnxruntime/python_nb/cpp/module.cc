// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Native part of the onnxruntime_nb package: bindings of the public C and C++ API.
// Conveniences, defaults, and exception classes live in the Python package.

#include <exception>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "adapter.h"
#include "devices.h"
#include "onnxruntime_cxx_api.h"
#include "model_package_bindings.h"
#include "session.h"
#include "value.h"

namespace nb = nanobind;
using namespace onnxruntime_nb;

namespace {

Ort::Env* g_env = nullptr;
std::optional<std::pair<int, int>> g_global_thread_pool_sizes;

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

}  // namespace

namespace onnxruntime_nb {

Ort::Env& GetEnv() {
  // Never destroyed: Python may release objects that use the environment after static destructors ran.
  if (g_env == nullptr) {
    if (g_global_thread_pool_sizes) {
      Ort::ThreadingOptions threading;
      threading.SetGlobalIntraOpNumThreads(g_global_thread_pool_sizes->first);
      threading.SetGlobalInterOpNumThreads(g_global_thread_pool_sizes->second);
      g_env = new Ort::Env(threading, ORT_LOGGING_LEVEL_WARNING, "onnxruntime");
    } else {
      g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime");
    }
  }
  return *g_env;
}

std::map<std::string, std::string> ToDict(const OrtKeyValuePairs* pairs) {
  if (pairs == nullptr) {
    return {};
  }
  auto map = Ort::ConstKeyValuePairs{pairs}.GetKeyValuePairs();
  return {map.begin(), map.end()};
}

void BindDevices(nb::module_& m) {
  nb::class_<HardwareDevice>(m, "OrtHardwareDevice")
      .def_prop_ro("type",
                   [](const HardwareDevice& d) { return PyEnum("OrtHardwareDeviceType", Ort::ConstHardwareDevice{d.ptr}.Type()); })
      .def_prop_ro("vendor_id", [](const HardwareDevice& d) { return Ort::ConstHardwareDevice{d.ptr}.VendorId(); })
      .def_prop_ro("vendor", [](const HardwareDevice& d) { return std::string(Ort::ConstHardwareDevice{d.ptr}.Vendor()); })
      .def_prop_ro("device_id", [](const HardwareDevice& d) { return Ort::ConstHardwareDevice{d.ptr}.DeviceId(); })
      .def_prop_ro("metadata",
                   [](const HardwareDevice& d) { return ToDict(Ort::ConstHardwareDevice{d.ptr}.Metadata()); })
      .def("__eq__", [](const HardwareDevice& a, const HardwareDevice& b) { return a.ptr == b.ptr; })
      .def("__hash__", [](const HardwareDevice& d) { return reinterpret_cast<uintptr_t>(d.ptr); });

  nb::class_<Ort::SyncStream>(m, "OrtSyncStream")
      .def("get_handle", [](Ort::SyncStream& s) { return reinterpret_cast<uintptr_t>(s.GetHandle()); });

  nb::class_<EpDevice>(m, "OrtEpDevice")
      .def_prop_ro("ep_name", [](const EpDevice& d) { return std::string(Ort::ConstEpDevice{d.ptr}.EpName()); })
      .def_prop_ro("ep_vendor", [](const EpDevice& d) { return std::string(Ort::ConstEpDevice{d.ptr}.EpVendor()); })
      .def_prop_ro("ep_metadata", [](const EpDevice& d) { return ToDict(Ort::ConstEpDevice{d.ptr}.EpMetadata()); })
      .def_prop_ro("ep_options", [](const EpDevice& d) { return ToDict(Ort::ConstEpDevice{d.ptr}.EpOptions()); })
      .def_prop_ro("device", [](const EpDevice& d) { return HardwareDevice{Ort::ConstEpDevice{d.ptr}.Device()}; })
      .def(
          "memory_info",
          [](nb::handle self, int memory_type) -> nb::object {
            auto info = Ort::ConstEpDevice{nb::cast<EpDevice&>(self).ptr}.GetMemoryInfo(
                static_cast<OrtDeviceMemoryType>(memory_type));
            if (info == nullptr) {
              return nb::none();
            }
            return nb::cast(MemoryInfo(info, nb::borrow(self)));
          },
          nb::arg("memory_type"))
      .def("create_sync_stream", [](const EpDevice& d) { return Ort::ConstEpDevice{d.ptr}.CreateSyncStream(); })
      .def("__eq__", [](const EpDevice& a, const EpDevice& b) { return a.ptr == b.ptr; })
      .def("__hash__", [](const EpDevice& d) { return reinterpret_cast<uintptr_t>(d.ptr); });

  m.def("get_ep_devices", []() {
    std::vector<EpDevice> devices;
    for (const auto& d : GetEnv().GetEpDevices()) {
      devices.push_back({d});
    }
    return devices;
  });
  m.def("get_hardware_devices", []() {
    std::vector<HardwareDevice> devices;
    for (const auto& d : GetEnv().GetHardwareDevices()) {
      devices.push_back({d});
    }
    return devices;
  });
  m.def(
      "get_hardware_device_ep_incompatibility_details",
      [](const std::string& ep_name, const HardwareDevice& device) {
        auto details = GetEnv().GetHardwareDeviceEpIncompatibilityDetails(ep_name.c_str(), Ort::ConstHardwareDevice{device.ptr});
        nb::dict result;
        result["reasons_bitmask"] = details.GetReasonsBitmask();
        const char* notes = details.GetNotes();
        result["notes"] = notes != nullptr && notes[0] != '\0' ? nb::object(nb::str(notes)) : nb::none();
        result["error_code"] = details.GetErrorCode();
        return result;
      },
      nb::arg("ep_name"), nb::arg("device"));
  m.def(
      "register_execution_provider_library",
      [](const std::string& name, const std::filesystem::path& path) {
        GetEnv().RegisterExecutionProviderLibrary(name.c_str(), path.native());
      },
      nb::arg("registration_name"), nb::arg("library_path"));
  m.def(
      "unregister_execution_provider_library",
      [](const std::string& name) { GetEnv().UnregisterExecutionProviderLibrary(name.c_str()); },
      nb::arg("registration_name"));
  m.def(
      "get_model_compatibility_for_ep_devices",
      [](const std::vector<EpDevice>& devices, const std::string& compatibility_info) {
        std::vector<const OrtEpDevice*> raw;
        for (const auto& d : devices) {
          raw.push_back(d.ptr);
        }
        OrtCompiledModelCompatibility status = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
        Ort::ThrowOnError(Ort::GetApi().GetModelCompatibilityForEpDevices(raw.data(), raw.size(),
                                                                          compatibility_info.c_str(), &status));
        return static_cast<int>(status);
      },
      nb::arg("ep_devices"), nb::arg("compatibility_info"));
  m.def(
      "get_compatibility_info_from_model",
      [](const std::filesystem::path& model_path, const std::string& ep_type) -> std::optional<std::string> {
        Ort::AllocatorWithDefaultOptions allocator;
        auto info = Ort::GetCompatibilityInfoFromModelAllocated(model_path.c_str(), ep_type.c_str(), allocator);
        if (info.get() == nullptr) {
          return std::nullopt;
        }
        return std::string(info.get());
      },
      nb::arg("model_path"), nb::arg("ep_type"));
  m.def(
      "get_compatibility_info_from_model_bytes",
      [](nb::bytes model, const std::string& ep_type) -> std::optional<std::string> {
        Ort::AllocatorWithDefaultOptions allocator;
        auto info =
            Ort::GetCompatibilityInfoFromModelBytesAllocated(model.data(), model.size(), ep_type.c_str(), allocator);
        if (info.get() == nullptr) {
          return std::nullopt;
        }
        return std::string(info.get());
      },
      nb::arg("model_data"), nb::arg("ep_type"));
  m.def(
      "copy_tensors",
      [](const std::vector<const Value*>& src, const std::vector<Value*>& dst, Ort::SyncStream* stream) {
        if (src.size() != dst.size()) {
          throw Ort::Exception("the number of sources and destinations differ", ORT_INVALID_ARGUMENT);
        }
        std::vector<const OrtValue*> raw_src;
        std::vector<OrtValue*> raw_dst;
        for (size_t i = 0; i < src.size(); ++i) {
          raw_src.push_back(src[i]->value);
          raw_dst.push_back(dst[i]->value);
        }
        Ort::ThrowOnError(Ort::GetApi().CopyTensors(GetEnv(), raw_src.data(), raw_dst.data(),
                                                    stream ? static_cast<OrtSyncStream*>(*stream) : nullptr,
                                                    raw_src.size()));
      },
      nb::arg("src"), nb::arg("dst"), nb::arg("stream").none());
}

}  // namespace onnxruntime_nb

NB_MODULE(_C, m) {
  RegisterExceptionTranslator();

  m.def("get_available_providers", &Ort::GetAvailableProviders);
  m.def("get_all_providers", &Ort::GetAllProviders);
  m.def("get_version_string", &Ort::GetVersionString);
  m.def("get_build_info", &Ort::GetBuildInfoString);
  m.def("get_device", &Ort::GetBuildDeviceString);
  m.def("set_seed", &Ort::SetGlobalRandomSeed, nb::arg("seed"));
  m.def(
      "set_default_logger_severity",
      [](int level) { GetEnv().UpdateEnvWithCustomLogLevel(static_cast<OrtLoggingLevel>(level)); },
      nb::arg("severity"));
  m.def(
      "set_default_logger_verbosity", [](int level) { GetEnv().UpdateEnvWithCustomLogVerbosityLevel(level); },
      nb::arg("vlog_level"));
  m.def("enable_telemetry_events", []() { GetEnv().EnableTelemetryEvents(); });
  m.def("disable_telemetry_events", []() { GetEnv().DisableTelemetryEvents(); });
  m.def(
      "set_global_thread_pool_sizes",
      [](int intra_op_num_threads, int inter_op_num_threads) {
        if (g_env != nullptr) {
          throw Ort::Exception("the global thread pool sizes must be set before the environment is created",
                               ORT_FAIL);
        }
        g_global_thread_pool_sizes = {intra_op_num_threads, inter_op_num_threads};
      },
      nb::arg("intra_op_num_threads"), nb::arg("inter_op_num_threads"));
  m.def("has_global_thread_pools", []() { return g_global_thread_pool_sizes.has_value(); });
  m.def(
      "create_and_register_allocator",
      [](const MemoryInfo& memory_info, std::optional<std::unordered_map<std::string, size_t>> arena_config) {
        Ort::ArenaCfg arena{nullptr};
        if (arena_config) {
          arena = Ort::ArenaCfg(*arena_config);
        }
        GetEnv().CreateAndRegisterAllocator(memory_info.ptr, arena);
      },
      nb::arg("memory_info"), nb::arg("arena_config").none());
  m.def(
      "create_and_register_allocator_v2",
      [](const std::string& provider_type, const MemoryInfo& memory_info,
         const std::unordered_map<std::string, std::string>& options,
         std::optional<std::unordered_map<std::string, size_t>> arena_config) {
        Ort::ArenaCfg arena{nullptr};
        if (arena_config) {
          arena = Ort::ArenaCfg(*arena_config);
        }
        GetEnv().CreateAndRegisterAllocatorV2(provider_type, memory_info.ptr, options, arena);
      },
      nb::arg("provider_type"), nb::arg("memory_info"), nb::arg("options"), nb::arg("arena_config").none());

  nb::class_<Ort::LoraAdapter>(m, "LoraAdapter")
      .def_static(
          "load",
          [](const std::filesystem::path& path) { return Ort::LoraAdapter::CreateLoraAdapter(path.native(), nullptr); },
          nb::arg("path"));

  BindValue(m);
  BindDevices(m);
  BindSession(m);
  BindModelPackage(m);
  BindAdapterFormat(m);
}
