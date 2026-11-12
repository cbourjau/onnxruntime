# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""The process-wide ONNX Runtime environment and the functions that configure it."""

from __future__ import annotations

from . import _C

# The process-wide environment. It holds state that must persist
# between sessions.
_ENV: _C.Env | None = None


def env() -> _C.Env:
    """Return the process-wide environment, creating it on first use."""
    global _ENV  # noqa: PLW0603
    if _ENV is None:
        _ENV = _C.Env()
    return _ENV


def set_default_logger_severity(severity: int) -> None:
    """Set the severity of the default logger: 0 (verbose), 1 (info), 2 (warning), 3 (error), or 4 (fatal)."""
    if not 0 <= severity <= 4:
        raise RuntimeError("Invalid logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal")
    env().update_log_severity_level(severity)


def set_default_logger_verbosity(vlog_level: int) -> None:
    """Set the verbosity level of the default logger.

    Verbose messages are logged only if the severity is 0 (verbose), and only in debug builds of ONNX Runtime.
    Sessions and runs log with their own verbosity levels, set with `SessionOptions.log_verbosity_level` and the
    run options, so this level rarely has a visible effect.
    """
    env().update_log_verbosity_level(vlog_level)
