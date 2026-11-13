# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""LoRA adapters and their file format."""

from __future__ import annotations

import os
import pathlib

from .capi import _C
from ._value import OrtValue


class AdapterFormat:
    """The parameters and versions of a LoRA adapter, which can be written to and read from adapter files."""

    def __init__(self) -> None:
        self._format_version = 1
        self._adapter_version = 0
        self._model_version = 0
        self._parameters: dict[str, OrtValue] = {}

    @staticmethod
    def read_adapter(file_path: str | os.PathLike) -> AdapterFormat:
        """Return the adapter of the file `file_path`."""
        format_version, adapter_version, model_version, parameters = _C.deserialize_adapter(
            pathlib.Path(file_path).read_bytes()
        )
        adapter = AdapterFormat()
        adapter._format_version = format_version
        adapter._adapter_version = adapter_version
        adapter._model_version = model_version
        adapter._parameters = {name: OrtValue(value) for name, value in parameters}
        return adapter

    def export_adapter(self, file_path: str | os.PathLike) -> None:
        """Write the adapter to the file `file_path`."""
        data = _C.serialize_adapter(
            self._adapter_version, self._model_version, [(name, v._value) for name, v in self._parameters.items()]
        )
        pathlib.Path(file_path).write_bytes(data)

    def get_format_version(self) -> int:
        return self._format_version

    def set_adapter_version(self, adapter_version: int) -> None:
        self._adapter_version = adapter_version

    def get_adapter_version(self) -> int:
        return self._adapter_version

    def set_model_version(self, model_version: int) -> None:
        self._model_version = model_version

    def get_model_version(self) -> int:
        return self._model_version

    def set_parameters(self, params: dict[str, OrtValue]) -> None:
        """Set the parameters, a dict of tensors on the CPU."""
        self._parameters = dict(params)

    def get_parameters(self) -> dict[str, OrtValue]:
        return dict(self._parameters)


class LoraAdapter:
    """A LoRA adapter that runs activate with `RunOptions.add_active_adapter`."""

    def __init__(self) -> None:
        self._native: _C.LoraAdapter | None = None

    def Load(self, adapter_path: str | os.PathLike) -> None:  # noqa: N802
        """Load the adapter file `adapter_path`."""
        self._native = _C.LoraAdapter.load(os.fspath(adapter_path))
