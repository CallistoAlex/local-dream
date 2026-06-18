#ifndef INPUMODEL_HPP
#define INPUMODEL_HPP

#include <QnnSampleApp.hpp>

using NpuStatus = qnn::tools::sample_app::StatusCode;

// Common NPU inference interface shared by Qualcomm QNN and MediaTek LiteRT
// backends. Pipelines interact with INpuModel only, never with vendor-specific
// types directly.
class INpuModel {
 public:
  virtual ~INpuModel() = default;

  virtual NpuStatus enablePerformaceMode() = 0;

  // SD1.5
  virtual NpuStatus executeUnetGraphs(float *latents, int timestep,
                                      float *text_embedding,
                                      float *latents_pred) = 0;
  virtual NpuStatus executeVaeEncoderGraphs(float *pixel_values, float *mean,
                                            float *std) = 0;
  virtual NpuStatus executeVaeDecoderGraphs(float *latents,
                                            float *pixel_values) = 0;

  // SDXL
  virtual NpuStatus executeUnetGraphsSDXL(float *sample, int timestep,
                                            float *encoder_hidden_states,
                                            float *text_embeds, float *time_ids,
                                            float *out_sample) = 0;
  virtual NpuStatus executeVaeEncoderGraphsSDXL(float *pixel_values,
                                                float *mean, float *std) = 0;
  virtual NpuStatus executeVaeDecoderGraphsSDXL(float *latents,
                                                float *pixel_values) = 0;

  // Upscaler (Qualcomm only today; MTK returns FAILURE)
  virtual NpuStatus executeUpscalerGraphs(float *input_image,
                                          float *output_image) {
    (void)input_image;
    (void)output_image;
    return NpuStatus::FAILURE;
  }
};

#endif  // INPUMODEL_HPP
