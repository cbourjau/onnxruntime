# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Pytest plugin that runs the test suite of the pybind11 bindings against `onnxruntime_nb`.

It makes `import onnxruntime` import `onnxruntime_nb`. The pure-Python subpackages of `onnxruntime`, such as
`onnxruntime.backend`, are taken from the `onnxruntime` package that is importable from the working directory.
"""

import importlib
import importlib.util
import sys

import onnxruntime_nb

_spec = importlib.util.find_spec("onnxruntime")
if _spec is None or _spec.submodule_search_locations is None:
    raise RuntimeError("the 'onnxruntime' package must be importable to provide its pure-Python subpackages")
onnxruntime_nb.__path__.append(_spec.submodule_search_locations[0])
sys.modules["onnxruntime"] = onnxruntime_nb

# Private names that the test modules import at module level.
_OrtValueVector = type("OrtValueVector", (list,), {})
for _module in ("onnxruntime.capi.onnxruntime_pybind11_state", "onnxruntime.capi._pybind_state"):
    importlib.import_module(_module).OrtValueVector = _OrtValueVector
_collection = importlib.import_module("onnxruntime.capi.onnxruntime_inference_collection")
for _name in ("_GRAPH_ANNOTATION_SKIP", "_graph_annotation_id", "_has_foreign_webgpu_context"):
    setattr(_collection, _name, None)

# Tests that exercise private names of the pybind11 bindings, mapped to the reason they are skipped.
_PRIVATE = "uses private names of the pybind11 bindings"
SKIPPED_TESTS = {
    "onnxruntime_test_python.py::TestInferenceSession::test_device_ortvalue_provenance_rules": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_foreign_webgpu_context_predicate": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_graph_annotation_id_run_option": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_is_webgpu_buffer_is_read_only": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_ort_value": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_ort_value_dlpack": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_ort_value_dlpack_zero_size": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_ort_value_from_dlpack_rejects_invalid_capsules": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_ortvalue_ownership_allows_matching_webgpu_context": (
        _PRIVATE
    ),
    "onnxruntime_test_python.py::TestInferenceSession::test_reset_session_releases_captured_graphs": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_run_with_ortvaluevector_is_gated_only_on_capture": (
        _PRIVATE
    ),
    "onnxruntime_test_python.py::TestInferenceSession::test_session_scoped_cpu_ortvalue": _PRIVATE,
    "onnxruntime_test_python.py::TestInferenceSession::test_webgpu_context_id_reports_default_for_cpu_session": (
        _PRIVATE
    ),
    "onnxruntime_test_python_iobinding.py::TestIOBinding::test_bind_onnx_types_supported_by_numpy": _PRIVATE,
    "onnxruntime_test_python_sparse_matmul.py::TestSparseToDenseMatmul::test_run_sparse_output_ort_value_vector": (
        _PRIVATE
    ),
}


def pytest_collection_modifyitems(config, items):
    import pytest  # noqa: PLC0415

    for item in items:
        reason = SKIPPED_TESTS.get(item.nodeid.split("[")[0])
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
