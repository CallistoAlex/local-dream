#ifndef PIPELINENPUBASE_HPP
#define PIPELINENPUBASE_HPP

#include <memory>
#include <string>

#include "INpuModel.hpp"
#include "Pipeline.hpp"

// Shared base for NPU-backed formats (sd15npu/sd15mtk, sdxl/sdxlmtk): owns the
// three stage models and the cfg=1 uncond-skip capability common to per-half
// UNet execution.
class PipelineNpuBase : public Pipeline {
 public:
  using Pipeline::Pipeline;

 protected:
  bool canSkipUncond() const override { return true; }

  std::unique_ptr<INpuModel> unet_;
  std::unique_ptr<INpuModel> vae_decoder_;
  std::unique_ptr<INpuModel> vae_encoder_;
};

// Backward-compatible alias used throughout existing QNN pipelines.
using PipelineQnn = PipelineNpuBase;

#endif  // PIPELINENPUBASE_HPP
