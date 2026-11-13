# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Model packages: directories with the variants of models for different execution providers and devices."""

from __future__ import annotations

import enum
import os

from .capi import _C
from ._options import SessionOptions
from ._session import InferenceSession

#: Authoring of model packages; arguments and return values with JSON are strings.
ModelPackage = _C.ModelPackage


class ModelPackageWriteMode(enum.IntEnum):
    """How `ModelPackage.commit` writes components and executor information."""

    #: Keep inline and external entries as they are.
    PRESERVE = 0
    #: Write all external entries inline.
    DENSE = 1


class ModelPackageOptions:
    """Execution providers and options that select the variants of a model package."""

    def __init__(self, sess_options: SessionOptions) -> None:
        self._native = _C.ModelPackageOptions(sess_options._native_for_session(None))


class ModelPackageComponentContext:
    """The variant of a component of a model package that was selected for the execution providers."""

    def __init__(self, native: _C.ModelPackageComponentContext) -> None:
        self._native = native

    def get_selected_variant_name(self) -> str:
        """Return the name of the selected variant."""
        return self._native.selected_variant_name

    def get_selected_variant_folder_path(self) -> str:
        """Return the directory of the selected variant."""
        return os.fspath(self._native.selected_variant_folder_path)

    def create_session(self, sess_options: SessionOptions | None = None) -> InferenceSession:
        """Return a session of the selected variant.

        If `sess_options` is None, the session and provider options of the package apply; otherwise `sess_options`
        overrides them, except for path options of the package that `sess_options` does not set.
        """
        native_options = None if sess_options is None else sess_options._native_for_session(None)
        return InferenceSession._from_session(self._native.create_session(native_options), sess_options)


class ModelPackageContext:
    """A model package that is read to select variants of its components."""

    def __init__(self, package_root: str | os.PathLike) -> None:
        self._native = _C.ModelPackageContext(package_root)

    def get_schema_version(self) -> int:
        """Return the major version of the manifest schema."""
        return self._native.schema_version

    def get_component_count(self) -> int:
        """Return the number of components."""
        return len(self._native.component_names)

    def get_component_names(self) -> list[str]:
        """Return the names of the components."""
        return self._native.component_names

    def get_variant_count(self, component_name: str) -> int:
        """Return the number of variants of the component `component_name`."""
        return len(self._native.variant_names(component_name))

    def get_variant_names(self, component_name: str) -> list[str]:
        """Return the names of the variants of the component `component_name`."""
        return self._native.variant_names(component_name)

    def get_variant_ep_name(self, component_name: str, variant_name: str) -> str | None:
        """Return the execution provider of a variant, or None if it does not specify one."""
        return self._native.variant_ep_name(component_name, variant_name)

    def resolve_string_ref(self, string_ref: str, base_dir: str | None = None, must_exist: bool = False) -> str:
        """Return the path that `string_ref` refers to, relative to `base_dir` or the package root."""
        return self._native.resolve_string_ref(string_ref, base_dir, must_exist)

    def select_component(self, component_name: str, options: ModelPackageOptions) -> ModelPackageComponentContext:
        """Return the variant of the component `component_name` that suits the execution providers of `options`."""
        return ModelPackageComponentContext(self._native.select_component(component_name, options._native))
