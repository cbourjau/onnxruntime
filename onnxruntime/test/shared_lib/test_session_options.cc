// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>

#include "core/common/common.h"
#include "core/framework/config_options.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gmock/gmock.h"
#include "test/util/include/scoped_env_vars.h"

using namespace onnxruntime;

extern std::unique_ptr<Ort::Env> ort_env;

TEST(CApiTest, session_options_graph_optimization_level) {
  // Test set optimization level succeeds when valid level is provided.
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
}

TEST(CApiTest, session_options_deterministic_compute) {
  // Manual validation currently. Check that SetDeterministicCompute in abi_session_options.cc is hit.
  Ort::SessionOptions options;
  options.SetDeterministicCompute(true);
}

TEST(CApiTest, session_options_get_mem_pattern_enabled) {
  Ort::SessionOptions options;

  // Memory pattern is enabled by default
  ASSERT_TRUE(options.GetMemPatternEnabled());

  // Disable and verify
  options.DisableMemPattern();
  ASSERT_FALSE(options.GetMemPatternEnabled());

  // Re-enable and verify
  options.EnableMemPattern();
  ASSERT_TRUE(options.GetMemPatternEnabled());
}

TEST(CApiTest, session_options_get_execution_mode) {
  Ort::SessionOptions options;

  // Default is sequential
  ASSERT_EQ(options.GetExecutionMode(), ORT_SEQUENTIAL);

  // Set to parallel and verify
  options.SetExecutionMode(ORT_PARALLEL);
  ASSERT_EQ(options.GetExecutionMode(), ORT_PARALLEL);

  // Set back to sequential and verify
  options.SetExecutionMode(ORT_SEQUENTIAL);
  ASSERT_EQ(options.GetExecutionMode(), ORT_SEQUENTIAL);
}

TEST(CApiTest, session_options_getters_return_defaults) {
  Ort::SessionOptions options;
  EXPECT_EQ(options.GetIntraOpNumThreads(), 0);
  EXPECT_EQ(options.GetInterOpNumThreads(), 0);
  EXPECT_EQ(options.GetGraphOptimizationLevel(), ORT_ENABLE_ALL);
  EXPECT_FALSE(options.GetProfilingEnabled());
  EXPECT_EQ(options.GetProfileFilePrefix(), ORT_TSTR("onnxruntime_profile_"));
  EXPECT_EQ(options.GetOptimizedModelFilePath(), ORT_TSTR(""));
  EXPECT_TRUE(options.GetCpuMemArenaEnabled());
  EXPECT_EQ(options.GetLogId(), "");
  EXPECT_EQ(options.GetLogSeverityLevel(), -1);
  EXPECT_EQ(options.GetLogVerbosityLevel(), 0);
  EXPECT_TRUE(options.GetPerSessionThreadsEnabled());
  EXPECT_FALSE(options.GetDeterministicComputeEnabled());
}

TEST(CApiTest, session_options_getters_return_set_values) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(3);
  options.SetInterOpNumThreads(4);
  options.EnableProfiling(ORT_TSTR("prefix"));
  options.SetOptimizedModelFilePath(ORT_TSTR("optimized.onnx"));
  options.DisableCpuMemArena();
  options.SetLogId("log id");
  options.SetLogSeverityLevel(3);
  Ort::ThrowOnError(Ort::GetApi().SetSessionLogVerbosityLevel(options, 2));
  options.DisablePerSessionThreads();
  options.SetDeterministicCompute(true);

  EXPECT_EQ(options.GetIntraOpNumThreads(), 3);
  EXPECT_EQ(options.GetInterOpNumThreads(), 4);
  EXPECT_TRUE(options.GetProfilingEnabled());
  EXPECT_EQ(options.GetProfileFilePrefix(), ORT_TSTR("prefix"));
  EXPECT_EQ(options.GetOptimizedModelFilePath(), ORT_TSTR("optimized.onnx"));
  EXPECT_FALSE(options.GetCpuMemArenaEnabled());
  EXPECT_EQ(options.GetLogId(), "log id");
  EXPECT_EQ(options.GetLogSeverityLevel(), 3);
  EXPECT_EQ(options.GetLogVerbosityLevel(), 2);
  EXPECT_FALSE(options.GetPerSessionThreadsEnabled());
  EXPECT_TRUE(options.GetDeterministicComputeEnabled());

  for (auto level : {ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_LAYOUT, ORT_ENABLE_ALL}) {
    options.SetGraphOptimizationLevel(level);
    EXPECT_EQ(options.GetGraphOptimizationLevel(), level);
  }
}

TEST(CApiTest, session_get_session_options_returns_session_values) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(3);
  options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking, "1");
  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), options);

  Ort::SessionOptions session_options = session.GetSessionOptions();
  EXPECT_EQ(session_options.GetIntraOpNumThreads(), 3);
  EXPECT_EQ(session_options.GetConfigEntry(kOrtSessionOptionsConfigDisablePrepacking), "1");
}

#if !defined(ORT_MINIMAL_BUILD)
// With ORT_LOAD_CONFIG_FROM_MODEL=1, the session uses the options in the model instead of those it was created with.
TEST(CApiTest, session_get_session_options_returns_options_from_model) {
  const test::ScopedEnvironmentVariables env_vars{test::EnvVarMap{{"ORT_LOAD_CONFIG_FROM_MODEL", "1"}}};
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(3);
  options.SetLogId("log id");
  Ort::Session session(*ort_env, ORT_TSTR("testdata/model_with_valid_ort_config_json.onnx"), options);

  Ort::SessionOptions session_options = session.GetSessionOptions();
  EXPECT_EQ(session_options.GetInterOpNumThreads(), 5);
  EXPECT_EQ(session_options.GetIntraOpNumThreads(), 2);
  EXPECT_EQ(session_options.GetGraphOptimizationLevel(), ORT_ENABLE_ALL);
  EXPECT_TRUE(session_options.GetProfilingEnabled());
  EXPECT_EQ(session_options.GetLogId(), "");

  Ort::AllocatorWithDefaultOptions allocator;
  std::filesystem::remove(session.EndProfilingAllocated(allocator).get());
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD) && !defined(ORT_NO_EXCEPTIONS)

TEST(CApiTest, session_options_oversized_affinity_string) {
  Ort::SessionOptions options;
  std::string long_affinity_str(ConfigOptions::kMaxValueLength + 1, '0');
  try {
    options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, long_affinity_str.c_str());
    ASSERT_TRUE(false) << "Creation of config should have thrown exception";
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Config value is longer than maximum length: "));
  }
}

#endif

#if defined(USE_OPENVINO_PROVIDER_INTERFACE)
// Test that loading OpenVINO EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_openvino) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kOpenVINOExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_OPENVINO_PROVIDER_INTERFACE)

#if defined(USE_CUDA_PROVIDER_INTERFACE)
// Test that loading CUDA EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_cuda) {
  Ort::SessionOptions session_options;

  Ort::CUDAProviderOptions cuda_options;
  try {
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    FAIL() << "Appending CUDA options have thrown exception";
  } catch (const Ort::Exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Failed to load"));
  }
}
#endif  // defined(USE_CUDA_PROVIDER_INTERFACE)

#if !defined(ORT_MINIMAL_BUILD)
// Test that appending the deprecated provider-bridge NV TensorRT RTX EP by name reports the plugin EP flow.
TEST(CApiTest, session_options_append_removed_nv_tensorrt_rtx_ep) {
  const OrtApi& api = Ort::GetApi();

  for (const char* provider_name : {"NvTensorRtRtx", kNvTensorRTRTXExecutionProvider}) {
    Ort::SessionOptions session_options;
    Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options, provider_name,
                                                                               nullptr, nullptr, 0)};
    ASSERT_FALSE(status.IsOK());
    EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Use the NVIDIA TensorRT RTX plugin EP"));
  }
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if defined(USE_TENSORRT_PROVIDER_INTERFACE)
// Test that loading TensorRT EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_tensorrt) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  OrtTensorRTProviderOptionsV2* trt_options = nullptr;
  Ort::Status status1 = Ort::Status{api.CreateTensorRTProviderOptions(&trt_options)};
  ASSERT_TRUE(status1.IsOK());

  Ort::Status status2 = Ort::Status{api.SessionOptionsAppendExecutionProvider_TensorRT_V2(session_options,
                                                                                          trt_options)};
  ASSERT_FALSE(status2.IsOK());
  EXPECT_EQ(status2.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status2.GetErrorMessage(), testing::HasSubstr("Failed to load"));

  api.ReleaseTensorRTProviderOptions(trt_options);
}
#endif  // defined(USE_TENSORRT_PROVIDER_INTERFACE)

#if defined(USE_VITISAI_PROVIDER_INTERFACE)
// Test that loading VitisAI EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_vitisai) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kVitisAIExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_VITISAI_PROVIDER_INTERFACE)

#if defined(USE_QNN_PROVIDER_INTERFACE)
// Test that loading QNN EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_qnn) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kQnnExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_QNN_PROVIDER_INTERFACE)
