cmake_minimum_required(VERSION 3.26)
project(onnxruntime_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "onnxruntime source tree")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Debug" CACHE PATH "directory containing the onnxruntime shared library")
set(FLATBUFFERS_INCLUDE_DIR "${ONNXRUNTIME_LIB_DIR}/_deps/flatbuffers-src/include" CACHE PATH
    "flatbuffers headers of the version that generated onnxruntime/lora/adapter_format/adapter_schema.fbs.h")

find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

# For development without a wheel, the package is assembled in ${PACKAGE_DIR}: the Python sources of the bindings
# and the native module.
set(PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")

nanobind_add_module(_C STABLE_ABI
  cpp/adapter.cc
  cpp/model_package_bindings.cc
  cpp/module.cc
  cpp/session.cc
  cpp/value.cc)
target_include_directories(_C PRIVATE
  "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
  "${ONNXRUNTIME_ROOT}/onnxruntime"
  "${ONNXRUNTIME_ROOT}/model_package/include"
  "${FLATBUFFERS_INCLUDE_DIR}")
find_library(MODEL_PACKAGE_LIB model_package PATHS "${ONNXRUNTIME_LIB_DIR}/model_package" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(_C PRIVATE "${ONNXRUNTIME_LIB}")
# The model package library authors packages; ONNX Runtime only reads them.
target_link_libraries(_C PRIVATE "${MODEL_PACKAGE_LIB}")
set_target_properties(_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PACKAGE_DIR}/capi")

add_custom_target(onnxruntime_python ALL
  COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/python/onnxruntime" "${PACKAGE_DIR}")

# Wheel contents besides the Python sources of the bindings, which scikit-build-core takes from python/onnxruntime.
# The layout mirrors the package of the pybind11 bindings; see cmake/onnxruntime_python.cmake.

# The native module finds the ONNX Runtime library next to itself.
if(APPLE)
  set_target_properties(_C PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(_C PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS _C LIBRARY DESTINATION onnxruntime/capi)
# The ONNX Runtime library is installed once, under the name that the native module loads (e.g. the SONAME on Linux),
# rather than with its chain of symbolic links, which the wheel would store as copies.
install(CODE "set(ONNXRUNTIME_LIB_DIR [[${ONNXRUNTIME_LIB_DIR}]])")
install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
    MODULES "$<TARGET_FILE:_C>"
    RESOLVED_DEPENDENCIES_VAR onnxruntime_libs
    PRE_INCLUDE_REGEXES "onnxruntime"
    PRE_EXCLUDE_REGEXES ".*"
    DIRECTORIES "${ONNXRUNTIME_LIB_DIR}")
  foreach(lib IN LISTS onnxruntime_libs)
    get_filename_component(name "${lib}" NAME)
    file(REAL_PATH "${lib}" target)
    file(COPY_FILE "${target}" "${CMAKE_INSTALL_PREFIX}/onnxruntime/capi/${name}")
  endforeach()
]])
# ONNX Runtime loads it at runtime for execution providers that are built as shared libraries.
file(GLOB providers_shared "${ONNXRUNTIME_LIB_DIR}/*onnxruntime_providers_shared${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(FILES ${providers_shared} DESTINATION onnxruntime/capi)

set(py "${ONNXRUNTIME_ROOT}/onnxruntime/python")
set(util "${ONNXRUNTIME_ROOT}/tools/python/util")

install(FILES
  "${ONNXRUNTIME_ROOT}/LICENSE"
  "${ONNXRUNTIME_ROOT}/ThirdPartyNotices.txt"
  "${ONNXRUNTIME_ROOT}/docs/Privacy.md"
  DESTINATION onnxruntime)
install(FILES
  "${py}/convert_npz_to_onnx_adapter.py"
  "${py}/onnxruntime_collect_build_info.py"
  DESTINATION onnxruntime/capi)

foreach(subpackage backend datasets)
  install(DIRECTORY "${py}/${subpackage}/" DESTINATION "onnxruntime/${subpackage}"
    FILES_MATCHING PATTERN "*.py" PATTERN "*.onnx" PATTERN "*.pb")
endforeach()

# Subdirectories that are not part of the package of the pybind11 bindings.
set(excluded_subdirs
  PATTERN "test" EXCLUDE
  PATTERN "torch_extensions" EXCLUDE
  PATTERN "torch_export_patches" EXCLUDE
  REGEX "/bart/utils$" EXCLUDE
  REGEX "/(transformers/models|quantization/execution_providers)/__init__\\.py$" EXCLUDE)

foreach(subpackage quantization transformers)
  install(DIRECTORY "${py}/tools/${subpackage}/" DESTINATION "onnxruntime/${subpackage}"
    FILES_MATCHING PATTERN "*.py" ${excluded_subdirs})
endforeach()

# 'onnxruntime.tools' combines modules of onnxruntime/python/tools and tools/python/util.
file(GLOB tools_srcs "${py}/tools/*.py")
list(FILTER tools_srcs EXCLUDE REGEX "/__init__\\.py$")
install(FILES ${tools_srcs}
  "${util}/check_onnx_model_mobile_usability.py"
  "${util}/convert_onnx_models_to_ort.py"
  "${util}/file_utils.py"
  "${util}/logger.py"
  "${util}/make_dynamic_shape_fixed.py"
  "${util}/onnx_model_utils.py"
  "${util}/optimize_onnx_model.py"
  "${util}/pytorch_export_helpers.py"
  "${util}/reduced_build_config_parser.py"
  "${util}/update_onnx_opset.py"
  "${ONNXRUNTIME_ROOT}/tools/python/remove_initializer_from_input.py"
  DESTINATION onnxruntime/tools)
file(READ "${py}/tools/__init__.py" tools_init)
file(READ "${util}/__init__append.py" tools_init_append)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tools/__init__.py" "${tools_init}${tools_init_append}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tools/__init__.py" DESTINATION onnxruntime/tools)

file(GLOB tools_qnn_srcs "${py}/tools/qnn/*.py")
install(FILES ${tools_qnn_srcs} DESTINATION onnxruntime/tools/qnn)
foreach(subpackage mobile_helpers ort_format_model qdq_helpers)
  install(DIRECTORY "${util}/${subpackage}/" DESTINATION "onnxruntime/tools/${subpackage}"
    FILES_MATCHING PATTERN "*.py" ${excluded_subdirs})
endforeach()
install(FILES
  "${ONNXRUNTIME_ROOT}/tools/ci_build/github/android/nnapi_supported_ops.md"
  "${ONNXRUNTIME_ROOT}/tools/ci_build/github/apple/coreml_supported_mlprogram_ops.md"
  "${ONNXRUNTIME_ROOT}/tools/ci_build/github/apple/coreml_supported_neuralnetwork_ops.md"
  DESTINATION onnxruntime/tools/mobile_helpers)
install(DIRECTORY "${ONNXRUNTIME_ROOT}/onnxruntime/core/flatbuffers/ort_flatbuffers_py/"
  DESTINATION onnxruntime/tools/ort_format_model/ort_flatbuffers_py
  FILES_MATCHING PATTERN "*.py")
