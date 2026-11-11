# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx
import onnxruntime_nb
import pytest
from onnx import TensorProto, helper

import onnxruntime


def save_model(path, nodes, inputs, outputs):
    onnx.save(helper.make_model(helper.make_graph(nodes, "g", inputs, outputs)), path)
    return path


def identity_model(tmp_path, elem_type):
    x, y = (helper.make_tensor_value_info(name, elem_type, None) for name in ("x", "y"))
    return save_model(tmp_path / "identity.onnx", [helper.make_node("Identity", ["x"], ["y"])], [x], [y])


@pytest.fixture
def model_path(tmp_path):
    """Model computing `sum = x + y` and `product = x * y` on float32 tensors of shape (N, 3)."""
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["N", 3]) for name in ("x", "y")]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["N", 3]) for name in ("sum", "product")]
    nodes = [helper.make_node("Add", ["x", "y"], ["sum"]), helper.make_node("Mul", ["x", "y"], ["product"])]
    return save_model(tmp_path / "model.onnx", nodes, inputs, outputs)


@pytest.fixture
def feeds():
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal((2, 3), dtype=np.float32) for name in ("x", "y")}


def run_both(path, output_names, feeds, **kwargs):
    """Return the outputs of the pybind11 and the nanobind bindings."""
    expected = onnxruntime.InferenceSession(path, **kwargs).run(output_names, feeds)
    actual = onnxruntime_nb.InferenceSession(path, **kwargs).run(output_names, feeds)
    return actual, expected


def assert_outputs_equal(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert isinstance(a, np.ndarray)
        assert a.dtype == e.dtype
        np.testing.assert_array_equal(a, e)


def test_get_available_providers_matches_pybind():
    assert onnxruntime_nb.get_available_providers() == onnxruntime.get_available_providers()


def test_names_match_pybind(model_path):
    nb_session = onnxruntime_nb.InferenceSession(model_path)
    pb_session = onnxruntime.InferenceSession(model_path)
    assert nb_session.input_names == [i.name for i in pb_session.get_inputs()]
    assert nb_session.output_names == [o.name for o in pb_session.get_outputs()]


@pytest.mark.parametrize("output_names", [None, [], ["product"], ["product", "sum"]])
def test_run_matches_pybind(model_path, feeds, output_names):
    assert_outputs_equal(*run_both(model_path, output_names, feeds))


@pytest.mark.parametrize(
    ("elem_type", "array"),
    [
        (TensorProto.FLOAT, np.array([[1.5, -2.0]], dtype=np.float32)),
        (TensorProto.DOUBLE, np.array([[1.5, -2.0]], dtype=np.float64)),
        (TensorProto.FLOAT16, np.array([[1.5, -2.0]], dtype=np.float16)),
        (TensorProto.INT8, np.array([[-1, 2]], dtype=np.int8)),
        (TensorProto.INT16, np.array([[-1, 2]], dtype=np.int16)),
        (TensorProto.INT32, np.array([[-1, 2]], dtype=np.int32)),
        (TensorProto.INT64, np.array([[-1, 2]], dtype=np.int64)),
        (TensorProto.UINT8, np.array([[1, 2]], dtype=np.uint8)),
        (TensorProto.UINT16, np.array([[1, 2]], dtype=np.uint16)),
        (TensorProto.UINT32, np.array([[1, 2]], dtype=np.uint32)),
        (TensorProto.UINT64, np.array([[1, 2]], dtype=np.uint64)),
        (TensorProto.BOOL, np.array([[True, False]])),
        (TensorProto.STRING, np.array([["a", "äb"]], dtype=object)),
        (TensorProto.STRING, np.array([["a", "äb"]])),
        (TensorProto.FLOAT, np.ones((0, 2), dtype=np.float32)),
        (TensorProto.FLOAT, np.array(3.0, dtype=np.float32)),
    ],
)
def test_identity_round_trip_matches_pybind(tmp_path, elem_type, array):
    assert_outputs_equal(*run_both(identity_model(tmp_path, elem_type), None, {"x": array}))


def test_bytes_input_becomes_str_output(tmp_path):
    # The pybind11 bindings fail to decode numpy bytes arrays, so the expected output is spelled out.
    session = onnxruntime_nb.InferenceSession(identity_model(tmp_path, TensorProto.STRING))
    (output,) = session.run(None, {"x": np.array([[b"a", b"bc"]])})
    assert output.dtype == object
    assert output.tolist() == [["a", "bc"]]


def test_non_contiguous_input(model_path, feeds):
    x = np.ones((4, 3), dtype=np.float32)[::2]
    assert_outputs_equal(*run_both(model_path, None, {**feeds, "x": x}))


def test_outputs_outlive_session(model_path, feeds):
    outputs = onnxruntime_nb.InferenceSession(model_path).run(None, feeds)
    np.testing.assert_array_equal(outputs[0], feeds["x"] + feeds["y"])


def _run_with(path, feeds):
    return lambda module: module.InferenceSession(path).run(None, feeds)


@pytest.mark.parametrize(
    "case",
    ["missing model", "invalid protobuf", "unknown input", "wrong dtype", "wrong shape"],
)
def test_errors_match_pybind(tmp_path, model_path, feeds, case):
    invalid_path = tmp_path / "invalid.onnx"
    invalid_path.write_bytes(b"not a model")
    call = {
        "missing model": lambda module: module.InferenceSession(tmp_path / "missing.onnx"),
        "invalid protobuf": lambda module: module.InferenceSession(invalid_path),
        "unknown input": _run_with(model_path, {**feeds, "z": feeds["x"]}),
        "wrong dtype": _run_with(model_path, {**feeds, "x": feeds["x"].astype(np.float64)}),
        "wrong shape": _run_with(model_path, {**feeds, "x": feeds["x"][:, :2]}),
    }[case]
    with pytest.raises(Exception) as expected:
        call(onnxruntime)
    with pytest.raises(Exception) as actual:
        call(onnxruntime_nb)
    assert type(actual.value).__name__ == type(expected.value).__name__
    assert type(actual.value) is getattr(onnxruntime_nb, type(expected.value).__name__)


def test_session_options_defaults_match_pybind():
    expected = onnxruntime.SessionOptions()
    actual = onnxruntime_nb.SessionOptions()
    for name in (
        "intra_op_num_threads",
        "inter_op_num_threads",
        "log_severity_level",
        "log_verbosity_level",
        "logid",
        "enable_profiling",
        "profile_file_prefix",
        "optimized_model_filepath",
        "enable_cpu_mem_arena",
        "enable_mem_pattern",
    ):
        assert getattr(actual, name) == getattr(expected, name), name
    for name in ("graph_optimization_level", "execution_mode"):
        assert getattr(actual, name).name == getattr(expected, name).name, name


def test_session_config_entries():
    options = onnxruntime_nb.SessionOptions()
    options.add_session_config_entry("session.disable_prepacking", "1")
    assert options.get_session_config_entry("session.disable_prepacking") == "1"
    with pytest.raises(RuntimeError):
        options.get_session_config_entry("missing")


def test_session_options_are_applied(tmp_path, model_path, feeds):
    options = onnxruntime_nb.SessionOptions()
    options.graph_optimization_level = onnxruntime_nb.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.optimized_model_filepath = str(tmp_path / "optimized.onnx")
    options.add_session_config_entry("session.disable_prepacking", "1")
    outputs = onnxruntime_nb.InferenceSession(model_path, options).run(None, feeds)
    np.testing.assert_array_equal(outputs[0], feeds["x"] + feeds["y"])
    assert (tmp_path / "optimized.onnx").exists()


def test_invalid_session_config_entry_raises(model_path):
    options = onnxruntime_nb.SessionOptions()
    options.add_session_config_entry("session.graph_optimizations_loop_level", "not an int")
    with pytest.raises(onnxruntime_nb.RuntimeException):
        onnxruntime_nb.InferenceSession(model_path, options)


@pytest.mark.parametrize(
    ("providers", "provider_options"),
    [
        (["CPUExecutionProvider"], None),
        (["CPUExecutionProvider"], [{}]),
        ([("CPUExecutionProvider", {})], None),
    ],
)
def test_cpu_provider(model_path, feeds, providers, provider_options):
    session = onnxruntime_nb.InferenceSession(model_path, providers=providers, provider_options=provider_options)
    np.testing.assert_array_equal(session.run(None, feeds)[0], feeds["x"] + feeds["y"])


def test_unknown_provider_raises(model_path):
    with pytest.raises(onnxruntime_nb.InvalidArgument, match="Unknown provider name"):
        onnxruntime_nb.InferenceSession(model_path, providers=["NoSuchExecutionProvider"])


def test_provider_options_length_mismatch_raises(model_path):
    with pytest.raises(ValueError, match="same length"):
        onnxruntime_nb.InferenceSession(model_path, providers=["CPUExecutionProvider"], provider_options=[{}, {}])
