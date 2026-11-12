// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Bindings of sessions and their options.

#pragma once

#include <cstddef>
#include <memory>

#include <nanobind/nanobind.h>

#include "onnxruntime_cxx_api.h"

namespace onnxruntime_nb {

// Python objects that native session options refer to.
struct SessionOptionsState {
  nanobind::object selection_delegate;
  nanobind::object ep_context_data_read_func;
  size_t ep_context_data_max_size = 0;
  nanobind::list keep_alive;
};

struct SessionOptions {
  Ort::SessionOptions options;
  std::shared_ptr<SessionOptionsState> state = std::make_shared<SessionOptionsState>();
};

struct Session {
  Ort::Session session{nullptr};
  // The state of the options the session was created with.
  std::shared_ptr<SessionOptionsState> options_state;
};

void BindSession(nanobind::module_& m);

}  // namespace onnxruntime_nb
