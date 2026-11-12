# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Exceptions raised for errors reported by ONNX Runtime, named after `OrtErrorCode`."""

# ruff: noqa: N818  # names match the pybind11 bindings


class Fail(RuntimeError):
    """Generic failure."""


class InvalidArgument(RuntimeError):
    """An argument, such as an input name, dtype, or shape, is invalid."""


class NoSuchFile(RuntimeError):
    """A file does not exist."""


class NoModel(RuntimeError):
    """No model is loaded."""


class EngineError(RuntimeError):
    """Error in the execution engine."""


class RuntimeException(RuntimeError):
    """Error raised while running or initializing a session."""


class InvalidProtobuf(RuntimeError):
    """A model is not a valid protobuf message."""


class ModelLoaded(RuntimeError):
    """A model is already loaded."""


class NotImplemented(RuntimeError):
    """A feature is not implemented."""


class InvalidGraph(RuntimeError):
    """The model graph is invalid."""


class EPFail(RuntimeError):
    """Error in an execution provider."""


class ModelLoadCanceled(RuntimeError):
    """Loading the model was canceled."""


class ModelRequiresCompilation(RuntimeError):
    """The model must be compiled before it can be run."""


class NotFound(RuntimeError):
    """A requested item was not found."""


class DeviceReset(RuntimeError):
    """A device was reset."""


# Values of `OrtErrorCode` in onnxruntime_c_api.h.
_BY_ERROR_CODE: dict[int, type[Exception]] = {
    1: Fail,
    2: InvalidArgument,
    3: NoSuchFile,
    4: NoModel,
    5: EngineError,
    6: RuntimeException,
    7: InvalidProtobuf,
    8: ModelLoaded,
    9: NotImplemented,
    10: InvalidGraph,
    11: EPFail,
    12: ModelLoadCanceled,
    13: ModelRequiresCompilation,
    14: NotFound,
    15: DeviceReset,
}


def exception_class(error_code: int) -> type[Exception]:
    """Return the exception class for an `OrtErrorCode` value."""
    return _BY_ERROR_CODE.get(error_code, RuntimeError)
