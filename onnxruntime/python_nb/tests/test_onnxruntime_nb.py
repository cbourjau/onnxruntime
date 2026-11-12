# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import numpy as np
import onnx
import onnxruntime_nb
import pytest
from onnx import TensorProto, helper

import onnxruntime


def save_model(path, nodes, inputs, outputs, opset=None):
    opset_imports = None if opset is None else [helper.make_opsetid("", opset)]
    onnx.save(helper.make_model(helper.make_graph(nodes, "g", inputs, outputs), opset_imports=opset_imports), path)
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


def test_node_args_match_pybind(model_path):
    nb_session = onnxruntime_nb.InferenceSession(model_path)
    pb_session = onnxruntime.InferenceSession(model_path)
    for getter in ("get_inputs", "get_outputs"):
        actual = [(a.name, a.type, a.shape) for a in getattr(nb_session, getter)()]
        assert actual == [(a.name, a.type, a.shape) for a in getattr(pb_session, getter)()]


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


def test_unknown_provider_falls_back_to_cpu(model_path):
    with pytest.warns(UserWarning, match="Unknown Provider Type: NoSuchExecutionProvider"):
        session = onnxruntime_nb.InferenceSession(model_path, providers=["NoSuchExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_provider_options_length_mismatch_raises(model_path):
    with pytest.raises(ValueError, match="same length"):
        onnxruntime_nb.InferenceSession(model_path, providers=["CPUExecutionProvider"], provider_options=[{}, {}])


@pytest.fixture
def random_model_path(tmp_path):
    """Model without inputs whose output `y` comes from a RandomUniform node without a `seed` attribute."""
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [5])
    node = helper.make_node("RandomUniform", [], ["y"], shape=[5])
    # The CPU execution provider implements RandomUniform up to opset 21.
    return save_model(tmp_path / "random.onnx", [node], [], [y], opset=21)


def random_output(module, path, seed):
    module.set_seed(seed)
    return module.InferenceSession(path).run(None, {})[0]


def test_set_seed_matches_pybind(random_model_path):
    expected = random_output(onnxruntime, random_model_path, 42)
    np.testing.assert_array_equal(random_output(onnxruntime_nb, random_model_path, 42), expected)


def test_set_seed_determines_output(random_model_path):
    first = random_output(onnxruntime_nb, random_model_path, 42)
    np.testing.assert_array_equal(random_output(onnxruntime_nb, random_model_path, 42), first)
    assert not np.array_equal(random_output(onnxruntime_nb, random_model_path, 43), first)


def test_get_all_providers_matches_pybind():
    assert onnxruntime_nb.get_all_providers() == onnxruntime.get_all_providers()


@pytest.mark.parametrize("severity", [-1, 5])
def test_invalid_logger_severity_raises(severity):
    with pytest.raises(RuntimeError):
        onnxruntime.set_default_logger_severity(severity)
    with pytest.raises(RuntimeError):
        onnxruntime_nb.set_default_logger_severity(severity)


@pytest.mark.parametrize(("severity", "logs_info"), [(1, True), (3, False)])
def test_logger_severity_filters_messages(model_path, capfd, severity, logs_info):
    onnxruntime_nb.set_default_logger_severity(severity)
    try:
        onnxruntime_nb.InferenceSession(model_path)
    finally:
        onnxruntime_nb.set_default_logger_severity(2)
    assert ("[I:onnxruntime:" in capfd.readouterr().err) == logs_info


def test_set_logger_verbosity():
    # Verbose messages of sessions and runs are filtered by their own options; no effect is observable here.
    onnxruntime_nb.set_default_logger_verbosity(1)
    onnxruntime_nb.set_default_logger_verbosity(0)


def test_get_device_matches_pybind():
    assert onnxruntime_nb.get_device() == onnxruntime.get_device()


@pytest.mark.parametrize("providers", [None, ["CPUExecutionProvider"]])
def test_providers_match_pybind(model_path, providers):
    expected = onnxruntime.InferenceSession(model_path, providers=providers)
    actual = onnxruntime_nb.InferenceSession(model_path, providers=providers)
    assert actual.get_providers() == expected.get_providers()
    assert actual.get_provider_options() == expected.get_provider_options()


def test_tuning_results_match_pybind(model_path):
    expected = onnxruntime.InferenceSession(model_path).get_tuning_results()
    assert onnxruntime_nb.InferenceSession(model_path).get_tuning_results() == expected


@pytest.mark.parametrize("module", [onnxruntime, onnxruntime_nb])
def test_set_tuning_results_of_unknown_provider(model_path, module):
    session = module.InferenceSession(model_path)
    results = [{"ep": "UnknownEP", "results": {}, "validators": {}}]
    session.set_tuning_results(results)
    with pytest.raises(RuntimeError, match="Cannot find execution provider UnknownEP"):
        session.set_tuning_results(results, error_on_invalid=True)


def test_run_with_ort_values(model_path, feeds):
    session = onnxruntime_nb.InferenceSession(model_path)
    inputs = {name: onnxruntime_nb.OrtValue.ortvalue_from_numpy(value) for name, value in feeds.items()}
    outputs = session.run_with_ort_values(["product"], inputs)
    assert all(isinstance(o, onnxruntime_nb.OrtValue) for o in outputs)
    np.testing.assert_array_equal(outputs[0].numpy(), feeds["x"] * feeds["y"])


@pytest.mark.parametrize("shape", [(3,), (2, 3), (0,), (0, 4), (4, 0)])
def test_dlpack_round_trip_shares_memory(shape):
    array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    value = onnxruntime_nb.OrtValue.ortvalue_from_numpy(array)
    for source in (value, value.__dlpack__()):
        imported = onnxruntime_nb.OrtValue.from_dlpack(source)
        assert imported.shape() == list(shape)
        if array.size:
            assert imported.data_ptr() == value.data_ptr()
    np.testing.assert_array_equal(np.from_dlpack(value), array)


class _InvalidDlpackProvider:
    def __dlpack__(self):
        return object()


@pytest.mark.parametrize("source", [None, object(), _InvalidDlpackProvider()])
def test_from_dlpack_rejects_invalid_sources(source):
    with pytest.raises(TypeError, match="valid DLPack capsule"):
        onnxruntime_nb.OrtValue.from_dlpack(source)


def test_from_dlpack_rejects_consumed_capsule():
    capsule = onnxruntime_nb.OrtValue.ortvalue_from_numpy(np.ones(2, dtype=np.float32)).__dlpack__()
    onnxruntime_nb.OrtValue.from_dlpack(capsule)
    with pytest.raises(TypeError, match="valid DLPack capsule"):
        onnxruntime_nb.OrtValue.from_dlpack(capsule)


@pytest.fixture
def sessions(model_path):
    """Two independent sessions of the model."""
    return onnxruntime_nb.InferenceSession(model_path), onnxruntime_nb.InferenceSession(model_path)


def _session_value(session):
    return session.create_ortvalue_from_shape_and_type([2, 3], np.float32, "cpu")


def test_session_value_is_usable_with_its_session(sessions, feeds):
    session, _ = sessions
    value = _session_value(session)
    outputs = session.run(["sum"], {"x": value, "y": feeds["y"]})
    np.testing.assert_array_equal(outputs[0], value.numpy() + feeds["y"])
    binding = session.io_binding()
    binding.bind_ortvalue_input("x", value)
    binding.bind_ortvalue_output("sum", _session_value(session))


@pytest.mark.parametrize(
    "use",
    [
        lambda session, value, feeds: session.run(None, {"x": value, "y": feeds["y"]}),
        lambda session, value, feeds: session.run_with_ort_values(
            None, {"x": value, "y": onnxruntime_nb.OrtValue.ortvalue_from_numpy(feeds["y"])}
        ),
        lambda session, value, feeds: session.io_binding().bind_ortvalue_input("x", value),
        lambda session, value, feeds: session.io_binding().bind_ortvalue_output("sum", value),
        lambda session, value, feeds: _session_value(session).update_inplace(value),
    ],
    ids=["run", "run_with_ort_values", "bind_input", "bind_output", "update_inplace"],
)
def test_session_value_is_rejected_by_other_sessions(sessions, feeds, use):
    session, other = sessions
    with pytest.raises(ValueError, match="session that created it|same session"):
        use(other, _session_value(session), feeds)


def test_binding_is_rejected_by_other_sessions(sessions):
    session, other = sessions
    with pytest.raises(ValueError, match="session that created it"):
        other.run_with_iobinding(session.io_binding())


def test_session_value_is_rejected_after_set_providers(model_path, feeds):
    session = onnxruntime_nb.InferenceSession(model_path)
    value = _session_value(session)
    binding = session.io_binding()
    session.set_providers(["CPUExecutionProvider"])
    with pytest.raises(ValueError, match="session that created it"):
        session.run(None, {"x": value, "y": feeds["y"]})
    with pytest.raises(ValueError, match="session that created it"):
        session.run_with_iobinding(binding)


def test_session_options_of_session_include_options_from_model(monkeypatch):
    monkeypatch.setenv("ORT_LOAD_CONFIG_FROM_MODEL", "1")
    options = onnxruntime_nb.SessionOptions()
    options.logid = "replaced by the defaults"
    path = Path(__file__).parents[2] / "test" / "testdata" / "model_with_valid_ort_config_json.onnx"
    session = onnxruntime_nb.InferenceSession(path, options)
    session_options = session.get_session_options()
    assert (session_options.inter_op_num_threads, session_options.intra_op_num_threads) == (5, 2)
    assert session_options.enable_profiling
    assert session_options.logid == ""
    Path(session.end_profiling()).unlink()


def test_profile_file_prefix_is_kept_while_profiling_is_disabled():
    options = onnxruntime_nb.SessionOptions()
    options.profile_file_prefix = "prefix"
    options.enable_profiling = True
    options.enable_profiling = False
    assert options.profile_file_prefix == "prefix"
