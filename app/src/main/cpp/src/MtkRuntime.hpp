#ifndef MTKRUNTIME_HPP
#define MTKRUNTIME_HPP

#include <dlfcn.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "Logger.hpp"

// Forward declaration — full definition in MtkModel.hpp
class MtkModel;

// LiteRT C API types (minimal subset for dynamic loading).
typedef int LiteRtStatus;
typedef struct LiteRtEnvironmentT* LiteRtEnvironment;
typedef struct LiteRtModelT* LiteRtModel;
typedef struct LiteRtCompiledModelT* LiteRtCompiledModel;
typedef struct LiteRtOptionsT* LiteRtOptions;
typedef struct LiteRtTensorBufferT* LiteRtTensorBuffer;
typedef struct LiteRtTensorBufferRequirementsT* LiteRtTensorBufferRequirements;

static constexpr LiteRtStatus kLiteRtStatusOk = 0;
static constexpr int kLiteRtHwAcceleratorNpu = 1 << 2;
static constexpr int kLiteRtEnvOptionTagDispatchLibraryDir = 1;
static constexpr int kLiteRtEnvOptionTagRuntimeLibraryDir = 22;

struct LiteRtAny {
  int type;
  union {
    const char* str_value;
    int int_value;
  };
};

struct LiteRtEnvOption {
  int tag;
  LiteRtAny value;
};

namespace mtk_runtime {

struct LiteRtApi {
  void* runtime_handle = nullptr;
  void* dispatch_handle = nullptr;

  LiteRtStatus (*LiteRtCreateEnvironment)(int, const LiteRtEnvOption*,
                                          LiteRtEnvironment*) = nullptr;
  void (*LiteRtDestroyEnvironment)(LiteRtEnvironment) = nullptr;
  LiteRtStatus (*LiteRtCreateModelFromFile)(LiteRtEnvironment, const char*,
                                            LiteRtModel*) = nullptr;
  void (*LiteRtDestroyModel)(LiteRtModel) = nullptr;
  LiteRtStatus (*LiteRtCreateOptions)(LiteRtOptions*) = nullptr;
  void (*LiteRtDestroyOptions)(LiteRtOptions) = nullptr;
  LiteRtStatus (*LiteRtSetOptionsHardwareAccelerators)(LiteRtOptions,
                                                       int) = nullptr;
  LiteRtStatus (*LiteRtCreateCompiledModel)(LiteRtEnvironment, LiteRtModel,
                                            LiteRtOptions,
                                            LiteRtCompiledModel*) = nullptr;
  void (*LiteRtDestroyCompiledModel)(LiteRtCompiledModel) = nullptr;
  LiteRtStatus (*LiteRtGetCompiledModelInputBufferRequirements)(
      LiteRtCompiledModel, size_t, size_t,
      LiteRtTensorBufferRequirements*) = nullptr;
  LiteRtStatus (*LiteRtGetCompiledModelOutputBufferRequirements)(
      LiteRtCompiledModel, size_t, size_t,
      LiteRtTensorBufferRequirements*) = nullptr;
  LiteRtStatus (*LiteRtCreateManagedTensorBufferFromRequirements)(
      LiteRtEnvironment, LiteRtTensorBufferRequirements,
      LiteRtTensorBuffer*) = nullptr;
  void (*LiteRtDestroyTensorBuffer)(LiteRtTensorBuffer) = nullptr;
  LiteRtStatus (*LiteRtLockTensorBuffer)(LiteRtTensorBuffer, void**, size_t*,
                                           int) = nullptr;
  LiteRtStatus (*LiteRtUnlockTensorBuffer)(LiteRtTensorBuffer) = nullptr;
  LiteRtStatus (*LiteRtRunCompiledModel)(LiteRtCompiledModel, size_t, size_t,
                                         LiteRtTensorBuffer*, size_t,
                                         LiteRtTensorBuffer*) = nullptr;
};

inline LiteRtApi g_api;
inline LiteRtEnvironment g_env = nullptr;
inline std::string g_lib_dir;
inline bool g_initialized = false;

template <typename T>
inline bool loadSym(void* handle, const char* name, T& out) {
  out = reinterpret_cast<T>(dlsym(handle, name));
  if (!out) {
    QNN_ERROR("LiteRT: failed to load symbol %s: %s", name, dlerror());
    return false;
  }
  return true;
}

inline bool init(const std::string& lib_dir) {
  if (g_initialized) return true;
  g_lib_dir = lib_dir;
  std::filesystem::path lib(lib_dir);

  g_api.runtime_handle = dlopen((lib / "libLiteRtRuntimeCApi.so").c_str(), RTLD_NOW);
  if (!g_api.runtime_handle) {
    QNN_ERROR("Failed to load libLiteRtRuntimeCApi.so: %s", dlerror());
    return false;
  }

  g_api.dispatch_handle =
      dlopen((lib / "libLiteRtDispatch_MediaTek.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!g_api.dispatch_handle) {
    QNN_WARN("Failed to load libLiteRtDispatch_MediaTek.so: %s", dlerror());
  }

#define LOAD(name) \
  if (!loadSym(g_api.runtime_handle, #name, g_api.name)) return false

  LOAD(LiteRtCreateEnvironment);
  LOAD(LiteRtDestroyEnvironment);
  LOAD(LiteRtCreateModelFromFile);
  LOAD(LiteRtDestroyModel);
  LOAD(LiteRtCreateOptions);
  LOAD(LiteRtDestroyOptions);
  LOAD(LiteRtSetOptionsHardwareAccelerators);
  LOAD(LiteRtCreateCompiledModel);
  LOAD(LiteRtDestroyCompiledModel);
  LOAD(LiteRtGetCompiledModelInputBufferRequirements);
  LOAD(LiteRtGetCompiledModelOutputBufferRequirements);
  LOAD(LiteRtCreateManagedTensorBufferFromRequirements);
  LOAD(LiteRtDestroyTensorBuffer);
  LOAD(LiteRtLockTensorBuffer);
  LOAD(LiteRtUnlockTensorBuffer);
  LOAD(LiteRtRunCompiledModel);

#undef LOAD

  LiteRtEnvOption options[2] = {};
  int num_options = 0;
  options[num_options].tag = kLiteRtEnvOptionTagDispatchLibraryDir;
  options[num_options].value.type = 0;
  options[num_options].value.str_value = lib_dir.c_str();
  num_options++;
  options[num_options].tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
  options[num_options].value.type = 0;
  options[num_options].value.str_value = lib_dir.c_str();
  num_options++;

  LiteRtStatus status =
      g_api.LiteRtCreateEnvironment(num_options, options, &g_env);
  if (status != kLiteRtStatusOk || !g_env) {
    QNN_ERROR("LiteRtCreateEnvironment failed: %d", status);
    return false;
  }

  g_initialized = true;
  QNN_INFO("MediaTek LiteRT runtime initialized from %s", lib_dir.c_str());
  return true;
}

inline void shutdown() {
  if (g_env && g_api.LiteRtDestroyEnvironment) {
    g_api.LiteRtDestroyEnvironment(g_env);
    g_env = nullptr;
  }
  if (g_api.dispatch_handle) {
    dlclose(g_api.dispatch_handle);
    g_api.dispatch_handle = nullptr;
  }
  if (g_api.runtime_handle) {
    dlclose(g_api.runtime_handle);
    g_api.runtime_handle = nullptr;
  }
  g_initialized = false;
}

inline std::unique_ptr<MtkModel> createModel(const std::string& modelPath,
                                             const std::string& modelName);

inline int initializeApp(const std::string& modelName,
                         std::unique_ptr<MtkModel>& app);

inline std::unique_ptr<MtkModel> createAndInitModel(
    const std::string& modelPath, const std::string& modelName);

}  // namespace mtk_runtime

#include "MtkRuntime.inl.hpp"

#endif  // MTKRUNTIME_HPP
