cmake_minimum_required(VERSION 3.26)
project(onnxruntime_nb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "onnxruntime source tree")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Debug" CACHE PATH "directory containing the onnxruntime shared library")

find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

# The package is assembled in ${PACKAGE_DIR}: the Python sources next to the native module.
set(PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_nb")

nanobind_add_module(_C STABLE_ABI
  cpp/module.cc
  cpp/ort_api.cc
  cpp/session.cc
  cpp/tensor.cc)
target_include_directories(_C PRIVATE "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
target_link_libraries(_C PRIVATE "${ONNXRUNTIME_LIB}")
set_target_properties(_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PACKAGE_DIR}")

add_custom_target(onnxruntime_nb_python ALL
  COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/python/onnxruntime_nb" "${PACKAGE_DIR}")
