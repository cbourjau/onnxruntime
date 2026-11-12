// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_package_bindings.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "model_package.h"
#include "session.h"
#include "value.h"

namespace nb = nanobind;

namespace onnxruntime_nb {

namespace {

// Raises the Python exception that corresponds to `status`, if any. Requires the GIL.
void CheckPackageStatus(ModelPackageStatus* status) {
  if (status == nullptr) {
    return;
  }
  std::unique_ptr<ModelPackageStatus, decltype(&ModelPackageStatus_Release)> owned{status,
                                                                                   ModelPackageStatus_Release};
  const std::string message = ModelPackageStatus_Message(owned.get());
  switch (ModelPackageStatus_Code(owned.get())) {
    case MODEL_PACKAGE_ERR_NOT_FOUND:
    case MODEL_PACKAGE_ERR_ASSET_MISSING:
      PyErr_SetString(PyExc_FileNotFoundError, message.c_str());
      throw nb::python_error();
    case MODEL_PACKAGE_ERR_IO:
      PyErr_SetString(PyExc_OSError, message.c_str());
      throw nb::python_error();
    case MODEL_PACKAGE_ERR_SCHEMA:
    case MODEL_PACKAGE_ERR_VERSION:
    case MODEL_PACKAGE_ERR_PATH_CONFINEMENT:
    case MODEL_PACKAGE_ERR_ASSET_HASH_MISMATCH:
    case MODEL_PACKAGE_ERR_INVALID_ARG:
      throw nb::value_error(message.c_str());
    default:
      throw std::runtime_error(message);
  }
}

using OptionalString = std::optional<std::string>;

const char* CStr(const OptionalString& s) { return s ? s->c_str() : nullptr; }

OptionalString Copy(const char* s) { return s ? OptionalString{s} : std::nullopt; }

// Copy of the info tree of a package. The tree belongs to the package and changes with it, so it is copied before
// any Python object is created; creating Python objects can run arbitrary code, e.g. finalizers.
struct PackageInfo {
  struct Variant {
    OptionalString name, variant_directory, ep, device, compatibility_string, additional_metadata_json;
    std::vector<std::pair<std::string, std::string>> executor_info;
  };
  struct Component {
    OptionalString name, additional_metadata_json;
    std::vector<Variant> variants;
  };
  struct SharedAsset {
    OptionalString uri, resolved_path;
  };

  int64_t schema_version_major{};
  int64_t schema_version_minor{};
  OptionalString package_name, package_version, description, layout, additional_metadata_json;
  std::vector<Component> components;
  std::vector<SharedAsset> shared_assets;

  explicit PackageInfo(const ModelPackageInfo& info)
      : schema_version_major(info.schema_version_major),
        schema_version_minor(info.schema_version_minor),
        package_name(Copy(info.package_name)),
        package_version(Copy(info.package_version)),
        description(Copy(info.description)),
        layout(Copy(info.layout)),
        additional_metadata_json(Copy(info.additional_metadata_json)) {
    for (size_t c = 0; c < info.num_components; ++c) {
      const ModelComponentInfo& component = info.components[c];
      Component& copy = components.emplace_back();
      copy.name = Copy(component.name);
      copy.additional_metadata_json = Copy(component.additional_metadata_json);
      for (size_t v = 0; v < component.num_variants; ++v) {
        const ModelVariantInfo& variant = component.variants[v];
        copy.variants.push_back({Copy(variant.name), Copy(variant.variant_directory), Copy(variant.ep), Copy(variant.device), Copy(variant.compatibility_string), Copy(variant.additional_metadata_json), {}});
        for (size_t e = 0; e < variant.num_executor_infos; ++e) {
          copy.variants.back().executor_info.emplace_back(variant.executor_infos[e].namespace_key,
                                                          variant.executor_infos[e].json);
        }
      }
    }
    for (size_t a = 0; a < info.num_shared_assets; ++a) {
      shared_assets.push_back({Copy(info.shared_assets[a].uri), Copy(info.shared_assets[a].resolved_path)});
    }
  }

  nb::dict ToDict() const {
    nb::list py_components;
    for (const auto& component : components) {
      nb::list py_variants;
      for (const auto& variant : component.variants) {
        nb::dict executor_info;
        for (const auto& [key, json] : variant.executor_info) {
          executor_info[nb::str(key.c_str())] = nb::str(json.c_str());
        }
        nb::dict d;
        d["name"] = variant.name;
        d["variant_directory"] = variant.variant_directory;
        d["ep"] = variant.ep;
        d["device"] = variant.device;
        d["compatibility_string"] = variant.compatibility_string;
        d["additional_metadata_json"] = variant.additional_metadata_json;
        d["executor_info"] = executor_info;
        py_variants.append(d);
      }
      nb::dict d;
      d["name"] = component.name;
      d["additional_metadata_json"] = component.additional_metadata_json;
      d["variants"] = py_variants;
      py_components.append(d);
    }
    nb::list py_assets;
    for (const auto& asset : shared_assets) {
      nb::dict d;
      d["uri"] = asset.uri;
      d["resolved_path"] = asset.resolved_path;
      py_assets.append(d);
    }
    nb::dict d;
    d["schema_version_major"] = schema_version_major;
    d["schema_version_minor"] = schema_version_minor;
    d["package_name"] = package_name;
    d["package_version"] = package_version;
    d["description"] = description;
    d["layout"] = layout;
    d["additional_metadata_json"] = additional_metadata_json;
    d["components"] = py_components;
    d["shared_assets"] = py_assets;
    return d;
  }
};

// A package of the model package library. All calls hold the GIL, which serializes them.
class Package {
 public:
  explicit Package(ModelPackage* package) : package_(package, ModelPackage_Close) {}

  ModelPackage* Get() const {
    if (!package_) {
      throw nb::value_error("model package is closed");
    }
    return package_.get();
  }

  void Close() { package_.reset(); }

 private:
  std::unique_ptr<ModelPackage, decltype(&ModelPackage_Close)> package_;
};

// Returns the string that a model package library function writes to `out`.
template <typename F>
std::string GetString(F&& f) {
  const char* out = nullptr;
  CheckPackageStatus(f(&out));
  return out;
}

void BindPackage(nb::module_& m) {
  nb::class_<Package>(m, "ModelPackage")
      .def(
          "__init__",
          [](Package* self) {
            ModelPackage* package = nullptr;
            CheckPackageStatus(ModelPackage_New(&package));
            new (self) Package(package);
          },
          "Create an empty package.")
      .def_static(
          "open",
          [](const std::string& root, bool allow_external_paths, bool follow_symlinks, bool strict_unknown_fields) {
            ModelPackageOpenOptions options{allow_external_paths, follow_symlinks, strict_unknown_fields};
            ModelPackage* package = nullptr;
            CheckPackageStatus(ModelPackage_Open(root.c_str(), &options, &package));
            return std::make_unique<Package>(package);
          },
          nb::arg("package_root"), nb::arg("allow_external_paths") = false, nb::arg("follow_symlinks") = true,
          nb::arg("strict_unknown_fields") = true, "Open the package in the directory `package_root`.")
      .def("close", &Package::Close, "Release the package; further calls raise ValueError.")
      .def("__enter__", [](Package& p) -> Package& {
             p.Get();
             return p; }, nb::rv_policy::reference)
      .def("__exit__", [](Package& p, nb::handle, nb::handle, nb::handle) { p.Close(); }, nb::arg().none(), nb::arg().none(), nb::arg().none())
      .def("set_metadata", [](Package& p, const OptionalString& name, const OptionalString& version, const OptionalString& description) { CheckPackageStatus(ModelPackage_SetMetadata(p.Get(), CStr(name), CStr(version), CStr(description))); }, nb::arg("name") = nb::none(), nb::arg("version") = nb::none(), nb::arg("description") = nb::none())
      .def("set_layout", [](Package& p, const std::string& layout) { CheckPackageStatus(ModelPackage_SetLayout(p.Get(), layout.c_str())); }, nb::arg("layout"))
      .def("set_component_inline", [](Package& p, const std::string& name, const std::string& json) { CheckPackageStatus(ModelPackage_SetComponentInline(p.Get(), name.c_str(), json.c_str())); }, nb::arg("name"), nb::arg("component_json"))
      .def("set_component_external", [](Package& p, const std::string& name, const std::string& path) { CheckPackageStatus(ModelPackage_SetComponentExternal(p.Get(), name.c_str(), path.c_str())); }, nb::arg("name"), nb::arg("path"))
      .def("remove_component", [](Package& p, const std::string& name) { CheckPackageStatus(ModelPackage_RemoveComponent(p.Get(), name.c_str())); }, nb::arg("name"))
      .def("set_variant", [](Package& p, const std::string& component, const std::string& variant, const std::string& json) { CheckPackageStatus(ModelPackage_SetVariant(p.Get(), component.c_str(), variant.c_str(), json.c_str())); }, nb::arg("component_name"), nb::arg("variant_name"), nb::arg("variant_json"))
      .def("remove_variant", [](Package& p, const std::string& component, const std::string& variant) { CheckPackageStatus(ModelPackage_RemoveVariant(p.Get(), component.c_str(), variant.c_str())); }, nb::arg("component_name"), nb::arg("variant_name"))
      .def("set_variant_executor_info_inline", [](Package& p, const std::string& component, const std::string& variant, const std::string& key, const std::string& json) { CheckPackageStatus(ModelPackage_SetVariantExecutorInfoInline(p.Get(), component.c_str(), variant.c_str(),
                                                                                                                                                                                                                                        key.c_str(), json.c_str())); }, nb::arg("component_name"), nb::arg("variant_name"), nb::arg("namespace_key"), nb::arg("info_json"))
      .def("set_variant_executor_info_external", [](Package& p, const std::string& component, const std::string& variant, const std::string& key, const std::string& path) { CheckPackageStatus(ModelPackage_SetVariantExecutorInfoExternal(p.Get(), component.c_str(),
                                                                                                                                                                                                                                            variant.c_str(), key.c_str(), path.c_str())); }, nb::arg("component_name"), nb::arg("variant_name"), nb::arg("namespace_key"), nb::arg("path"))
      .def("remove_variant_executor_info", [](Package& p, const std::string& component, const std::string& variant, const std::string& key) { CheckPackageStatus(
                                                                                                                                                  ModelPackage_RemoveVariantExecutorInfo(p.Get(), component.c_str(), variant.c_str(), key.c_str())); }, nb::arg("component_name"), nb::arg("variant_name"), nb::arg("namespace_key"))
      .def("set_additional_metadata_json", [](Package& p, const std::string& scope, const OptionalString& component, const OptionalString& variant, const OptionalString& json) { CheckPackageStatus(ModelPackage_SetAdditionalMetadataJson(p.Get(), scope.c_str(), CStr(component),
                                                                                                                                                                                                                                            CStr(variant), CStr(json))); }, nb::arg("scope"), nb::arg("component_name") = nb::none(), nb::arg("variant_name") = nb::none(), nb::arg("json") = nb::none())
      .def("add_shared_asset", [](Package& p, const std::string& source, const OptionalString& expected_uri, bool copy_in) { return GetString([&](const char** out) {
                                                                                                                               return ModelPackage_AddSharedAsset(p.Get(), source.c_str(), CStr(expected_uri), copy_in, out);
                                                                                                                             }); }, nb::arg("source_dir"), nb::arg("expected_uri") = nb::none(), nb::arg("copy_in") = true,
           "Add the directory `source_dir` as shared asset and return its URI. The directory must stay readable "
           "until the package is committed.")
      .def("remove_shared_asset", [](Package& p, const std::string& uri) { CheckPackageStatus(ModelPackage_RemoveSharedAsset(p.Get(), uri.c_str())); }, nb::arg("uri"))
      .def("get_info", [](const Package& p) { return PackageInfo(*ModelPackage_Info(p.Get())).ToDict(); }, "Return the metadata, components, variants, and shared assets of the package as a dict.")
      .def("get_component_json", [](const Package& p, const std::string& component) { return GetString(
                                                                                          [&](const char** out) { return ModelPackage_GetComponentJson(p.Get(), component.c_str(), out); }); }, nb::arg("component_name"))
      .def("get_variant_json", [](const Package& p, const std::string& component, const std::string& variant) { return GetString([&](const char** out) {
                                                                                                                  return ModelPackage_GetVariantJson(p.Get(), component.c_str(), variant.c_str(), out);
                                                                                                                }); }, nb::arg("component_name"), nb::arg("variant_name"))
      .def("resolve_asset_uri", [](const Package& p, const std::string& uri) { return GetString([&](const char** out) { return ModelPackage_ResolveAssetUri(p.Get(), uri.c_str(), out); }); }, nb::arg("uri"))
      .def("resolve_string_ref", [](const Package& p, const std::string& ref, const OptionalString& base, bool must_exist) { return GetString([&](const char** out) {
                                                                                                                               return ModelPackage_ResolveStringRef(p.Get(), CStr(base), ref.c_str(), must_exist, out);
                                                                                                                             }); }, nb::arg("string_ref"), nb::arg("base_dir") = nb::none(), nb::arg("must_exist") = false)
      .def_static("compute_directory_hash", [](const std::string& source) { return GetString([&](const char** out) { return ModelPackage_ComputeDirectoryHash(source.c_str(), out); }); }, nb::arg("source_dir"))
      .def("commit", [](Package& p, const OptionalString& destination, int mode) { CheckPackageStatus(
                                                                                       ModelPackage_Commit(p.Get(), CStr(destination), static_cast<ModelPackageWriteMode>(mode))); }, nb::arg("dest_root") = nb::none(), nb::arg("mode") = static_cast<int>(MODEL_PACKAGE_WRITE_PRESERVE), "Write the package in place, or to the empty directory `dest_root`.")
      .def("prune", [](Package& p) { CheckPackageStatus(ModelPackage_Prune(p.Get())); });
}

// ONNX Runtime's model package contexts cache data without synchronization; all calls hold the GIL, which
// serializes them.
void BindPackageContexts(nb::module_& m) {
  nb::class_<Ort::ModelPackageOptions>(m, "ModelPackageOptions")
      .def(
          "__init__",
          [](Ort::ModelPackageOptions* self, const SessionOptions& options) {
            new (self) Ort::ModelPackageOptions(GetEnv(), options.options);
          },
          nb::arg("options"));

  nb::class_<Ort::ModelPackageComponentContext>(m, "ModelPackageComponentContext")
      .def_prop_ro("selected_variant_name",
                   [](const Ort::ModelPackageComponentContext& c) { return c.GetSelectedVariantName(); })
      .def_prop_ro("selected_variant_folder_path",
                   [](const Ort::ModelPackageComponentContext& c) {
                     return std::filesystem::path(c.GetSelectedVariantFolderPath());
                   })
      .def(
          "create_session",
          [](Ort::ModelPackageComponentContext& c, const SessionOptions* options) {
            if (options == nullptr) {
              return Session{c.CreateSession(GetEnv()), std::make_shared<SessionOptionsState>()};
            }
            return Session{c.CreateSession(GetEnv(), options->options), options->state};
          },
          nb::arg("options").none(),
          "Return a session of the selected variant; None applies the options of the package.");

  nb::class_<Ort::ModelPackageContext>(m, "ModelPackageContext")
      .def(
          "__init__",
          [](Ort::ModelPackageContext* self, const std::filesystem::path& root) {
            new (self) Ort::ModelPackageContext(root.c_str());
          },
          nb::arg("package_root"))
      .def_prop_ro("schema_version", &Ort::ModelPackageContext::GetSchemaVersion)
      .def_prop_ro("component_names", &Ort::ModelPackageContext::GetComponentNames)
      .def(
          "variant_names",
          [](const Ort::ModelPackageContext& c, const std::string& component) {
            return c.GetVariantNames(component.c_str());
          },
          nb::arg("component_name"))
      .def(
          "variant_ep_name",
          [](const Ort::ModelPackageContext& c, const std::string& component,
             const std::string& variant) -> OptionalString {
            return Copy(c.GetVariantEpName(component.c_str(), variant.c_str()));
          },
          nb::arg("component_name"), nb::arg("variant_name"))
      .def(
          "resolve_string_ref",
          [](const Ort::ModelPackageContext& c, const std::string& ref, const OptionalString& base, bool must_exist) {
            return c.ResolveStringRef(CStr(base), ref.c_str(), must_exist);
          },
          nb::arg("string_ref"), nb::arg("base_dir").none(), nb::arg("must_exist"))
      .def(
          "select_component",
          [](const Ort::ModelPackageContext& c, const std::string& component,
             const Ort::ModelPackageOptions& options) { return c.SelectComponent(component.c_str(), options); },
          nb::arg("component_name"), nb::arg("options"));
}

}  // namespace

void BindModelPackage(nb::module_& m) {
  BindPackage(m);
  BindPackageContexts(m);
}

}  // namespace onnxruntime_nb
