#ifndef MTKMODEL_HPP
#define MTKMODEL_HPP

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "Config.hpp"
#include "INpuModel.hpp"
#include "Logger.hpp"
#include "SDUtils.hpp"

// Forward declarations for LiteRT handles (defined in MtkRuntime.hpp).
struct LiteRtEnvironmentT;
struct LiteRtModelT;
struct LiteRtCompiledModelT;
struct LiteRtOptionsT;
struct LiteRtTensorBufferT;
struct LiteRtTensorBufferRequirementsT;
typedef struct LiteRtEnvironmentT* LiteRtEnvironment;
typedef struct LiteRtModelT* LiteRtModel;
typedef struct LiteRtCompiledModelT* LiteRtCompiledModel;
typedef struct LiteRtOptionsT* LiteRtOptions;
typedef struct LiteRtTensorBufferT* LiteRtTensorBuffer;
typedef struct LiteRtTensorBufferRequirementsT* LiteRtTensorBufferRequirements;

namespace mtk_runtime {
struct LiteRtApi;
extern LiteRtApi g_api;
extern LiteRtEnvironment g_env;
}  // namespace mtk_runtime

static constexpr LiteRtStatus kLiteRtStatusOk = 0;
static constexpr int kLiteRtHwAcceleratorNpu = 1 << 2;

// MediaTek NPU model backed by LiteRT CompiledModel API.
// Loads AOT-compiled .litert binaries and runs inference on NeuroPilot NPU.
class MtkModel : public INpuModel {
 public:
  MtkModel(std::string modelPath, std::string modelName)
      : model_path_(std::move(modelPath)), model_name_(std::move(modelName)) {}

  ~MtkModel() override { release(); }

  bool initialize() {
    using namespace mtk_runtime;
    if (initialized_) return true;

    LiteRtStatus status = g_api.LiteRtCreateModelFromFile(
        g_env, model_path_.c_str(), &model_);
    if (status != kLiteRtStatusOk || !model_) {
      QNN_ERROR("Failed to load MTK model from %s", model_path_.c_str());
      return false;
    }

    LiteRtOptions options = nullptr;
    status = g_api.LiteRtCreateOptions(&options);
    if (status != kLiteRtStatusOk) {
      QNN_ERROR("LiteRtCreateOptions failed for %s", model_name_.c_str());
      return false;
    }

    status = g_api.LiteRtSetOptionsHardwareAccelerators(
        options, kLiteRtHwAcceleratorNpu);
    if (status != kLiteRtStatusOk) {
      QNN_WARN("NPU accelerator option failed, trying NPU|CPU fallback");
      g_api.LiteRtSetOptionsHardwareAccelerators(
          options, kLiteRtHwAcceleratorNpu | (1 << 0));
    }

    status = g_api.LiteRtCreateCompiledModel(g_env, model_, options,
                                             &compiled_model_);
    g_api.LiteRtDestroyOptions(options);

    if (status != kLiteRtStatusOk || !compiled_model_) {
      QNN_ERROR("LiteRtCreateCompiledModel failed for %s",
                model_name_.c_str());
      return false;
    }

    if (!setupBuffers()) {
      QNN_ERROR("Failed to setup tensor buffers for %s", model_name_.c_str());
      return false;
    }

    initialized_ = true;
    return true;
  }

  NpuStatus enablePerformaceMode() override {
    // NeuroPilot performance hints are applied at compile time for AOT models.
    return NpuStatus::SUCCESS;
  }

  NpuStatus executeUnetGraphs(float *latents, int timestep,
                              float *text_embedding,
                              float *latents_pred) override {
    return runGraph(
        [&](float *in0, float *in1, float *in2) {
          int elementCount = 1 * 4 * sample_width * sample_height;
          memcpy(in0, latents, elementCount * sizeof(float));
          *reinterpret_cast<int32_t *>(in1) = timestep;
          int textCount = 1 * 77 * text_embedding_size;
          memcpy(in2, text_embedding, textCount * sizeof(float));
        },
        [&](float *out0) {
          int elementCount = 1 * 4 * sample_width * sample_height;
          memcpy(latents_pred, out0, elementCount * sizeof(float));
        });
  }

  NpuStatus executeVaeEncoderGraphs(float *pixel_values, float *mean,
                                    float *std) override {
    return runGraph(
        [&](float *in0, float *, float *) {
          int elementCount = 1 * 3 * output_width * output_height;
          memcpy(in0, pixel_values, elementCount * sizeof(float));
        },
        [&](float *out0) {
          int latentCount = 1 * 4 * sample_width * sample_height;
          memcpy(mean, out0, latentCount * sizeof(float));
          if (num_outputs_ > 1) {
            float *out1 = getOutputPtr(1);
            memcpy(std, out1, latentCount * sizeof(float));
          }
        });
  }

  NpuStatus executeVaeDecoderGraphs(float *latents, float *pixel_values) override {
    return runGraph(
        [&](float *in0, float *, float *) {
          int elementCount = 1 * 4 * sample_width * sample_height;
          memcpy(in0, latents, elementCount * sizeof(float));
        },
        [&](float *out0) {
          int elementCount = 1 * 3 * output_width * output_height;
          memcpy(pixel_values, out0, elementCount * sizeof(float));
        });
  }

  NpuStatus executeUnetGraphsSDXL(float *sample, int timestep,
                                  float *encoder_hidden_states,
                                  float *text_embeds, float *time_ids,
                                  float *out_sample) override {
    return runGraph(
        [&](float *in0, float *in1, float *in2) {
          int latentCount = 1 * 4 * sample_width * sample_height;
          memcpy(in0, sample, latentCount * sizeof(float));
          int textCount = 1 * 77 * (text_embedding_size + text_embedding_size_2);
          memcpy(in1, encoder_hidden_states, textCount * sizeof(float));
          *reinterpret_cast<int32_t *>(in2) = timestep;
          if (num_inputs_ > 3) {
            float *in3 = getInputPtr(3);
            memcpy(in3, time_ids, 6 * sizeof(float));
          }
          if (num_inputs_ > 4) {
            float *in4 = getInputPtr(4);
            memcpy(in4, text_embeds,
                   text_embedding_size_2 * sizeof(float));
          }
        },
        [&](float *out0) {
          int latentCount = 1 * 4 * sample_width * sample_height;
          memcpy(out_sample, out0, latentCount * sizeof(float));
        });
  }

  NpuStatus executeVaeEncoderGraphsSDXL(float *pixel_values, float *mean,
                                        float *std) override {
    return executeVaeEncoderGraphs(pixel_values, mean, std);
  }

  NpuStatus executeVaeDecoderGraphsSDXL(float *latents,
                                        float *pixel_values) override {
    return executeVaeDecoderGraphs(latents, pixel_values);
  }

 private:
  void release() {
    using namespace mtk_runtime;
    input_buffers_.clear();
    output_buffers_.clear();
    if (compiled_model_ && g_api.LiteRtDestroyCompiledModel) {
      g_api.LiteRtDestroyCompiledModel(compiled_model_);
      compiled_model_ = nullptr;
    }
    if (model_ && g_api.LiteRtDestroyModel) {
      g_api.LiteRtDestroyModel(model_);
      model_ = nullptr;
    }
    initialized_ = false;
  }

  bool setupBuffers() {
    using namespace mtk_runtime;
    num_inputs_ = 0;
    num_outputs_ = 0;

    // Probe inputs until requirements call fails.
    for (size_t i = 0; i < 16; ++i) {
      LiteRtTensorBufferRequirements req = nullptr;
      if (g_api.LiteRtGetCompiledModelInputBufferRequirements(
              compiled_model_, 0, i, &req) != kLiteRtStatusOk)
        break;
      LiteRtTensorBuffer buf = nullptr;
      if (g_api.LiteRtCreateManagedTensorBufferFromRequirements(g_env, req,
                                                                &buf) !=
          kLiteRtStatusOk)
        return false;
      input_buffers_.push_back(buf);
      num_inputs_++;
    }

    for (size_t i = 0; i < 16; ++i) {
      LiteRtTensorBufferRequirements req = nullptr;
      if (g_api.LiteRtGetCompiledModelOutputBufferRequirements(
              compiled_model_, 0, i, &req) != kLiteRtStatusOk)
        break;
      LiteRtTensorBuffer buf = nullptr;
      if (g_api.LiteRtCreateManagedTensorBufferFromRequirements(g_env, req,
                                                                &buf) !=
          kLiteRtStatusOk)
        return false;
      output_buffers_.push_back(buf);
      num_outputs_++;
    }

    QNN_INFO("MTK model %s: %zu inputs, %zu outputs", model_name_.c_str(),
             num_inputs_, num_outputs_);
    return num_inputs_ > 0 && num_outputs_ > 0;
  }

  float *getInputPtr(size_t index) {
    return lockBuffer(input_buffers_[index]);
  }

  float *getOutputPtr(size_t index) {
    return lockBuffer(output_buffers_[index]);
  }

  float *lockBuffer(LiteRtTensorBuffer buf) {
    using namespace mtk_runtime;
    void *ptr = nullptr;
    size_t size = 0;
    if (g_api.LiteRtLockTensorBuffer(buf, &ptr, &size, 2) != kLiteRtStatusOk)
      return nullptr;
    locked_buffers_.push_back(buf);
    return static_cast<float *>(ptr);
  }

  void unlockAll() {
    using namespace mtk_runtime;
    for (auto buf : locked_buffers_) {
      g_api.LiteRtUnlockTensorBuffer(buf);
    }
    locked_buffers_.clear();
  }

  template <typename FillInputs, typename ReadOutputs>
  NpuStatus runGraph(FillInputs fill_inputs, ReadOutputs read_outputs) {
    using namespace mtk_runtime;
    if (!initialized_) return NpuStatus::FAILURE;

    locked_buffers_.clear();
    float *in0 = num_inputs_ > 0 ? getInputPtr(0) : nullptr;
    float *in1 = num_inputs_ > 1 ? getInputPtr(1) : nullptr;
    float *in2 = num_inputs_ > 2 ? getInputPtr(2) : nullptr;
    fill_inputs(in0, in1, in2);
    unlockAll();

    auto start = std::chrono::high_resolution_clock::now();
    LiteRtStatus status = g_api.LiteRtRunCompiledModel(
        compiled_model_, 0, num_inputs_, input_buffers_.data(), num_outputs_,
        output_buffers_.data());
    auto end = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - start)
                       .count();
    QNN_INFO("%s graph execution time: %d ms", model_name_.c_str(), duration);

    if (status != kLiteRtStatusOk) {
      QNN_ERROR("%s graph execution failed!", model_name_.c_str());
      return NpuStatus::FAILURE;
    }

    locked_buffers_.clear();
    float *out0 = getOutputPtr(0);
    read_outputs(out0);
    unlockAll();
    return NpuStatus::SUCCESS;
  }

  const std::string model_path_;
  const std::string model_name_;
  LiteRtModel model_ = nullptr;
  LiteRtCompiledModel compiled_model_ = nullptr;
  std::vector<LiteRtTensorBuffer> input_buffers_;
  std::vector<LiteRtTensorBuffer> output_buffers_;
  std::vector<LiteRtTensorBuffer> locked_buffers_;
  size_t num_inputs_ = 0;
  size_t num_outputs_ = 0;
  bool initialized_ = false;
};

#endif  // MTKMODEL_HPP
