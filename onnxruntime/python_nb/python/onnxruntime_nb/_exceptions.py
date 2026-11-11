# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Exceptions raised for errors reported by ONNX Runtime, named as in the pybind11 bindings."""

# ruff: noqa: N818  # names match the pybind11 bindings


class Fail(Exception):
    """Generic failure."""


class InvalidArgument(Exception):
    """An argument, such as an input name, dtype, or shape, is invalid."""


class NoSuchFile(Exception):
    """A file does not exist."""


class NoModel(Exception):
    """No model is loaded."""


class EngineError(Exception):
    """Error in the execution engine."""


class RuntimeException(Exception):
    """Error raised while running or initializing a session."""


class InvalidProtobuf(Exception):
    """A model is not a valid protobuf message."""


class ModelLoaded(Exception):
    """A model is already loaded."""


class NotImplemented(Exception):
    """A feature is not implemented."""


class InvalidGraph(Exception):
    """The model graph is invalid."""


class EPFail(Exception):
    """Error in an execution provider."""


class ModelLoadCanceled(Exception):
    """Loading the model was canceled."""


class ModelRequiresCompilation(Exception):
    """The model must be compiled before it can be run."""


class NotFound(Exception):
    """A requested item was not found."""


class DeviceReset(Exception):
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
