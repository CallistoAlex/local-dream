#ifndef PIPELINESD15MTK_HPP
#define PIPELINESD15MTK_HPP

#include <MNN/Interpreter.hpp>
#include <memory>
#include <stdexcept>
#include <string>

#include "Config.hpp"
#include "MnnUtils.hpp"
#include "MtkRuntime.hpp"
#include "PipelineNpuBase.hpp"

// sd15mtk: UNet and VAE run on MediaTek NPU via LiteRT; CLIP on MNN CPU.
// Fixed 512px resolution (no zstd patches in v1).
class PipelineSd15Mtk : public PipelineNpuBase {
 public:
  PipelineSd15Mtk(TextEncoder &text_encoder, const std::string &model_dir,
                  std::string clip_path, std::string unet_path,
                  std::string vae_decoder_path, std::string vae_encoder_path,
                  bool use_v_pred)
      : PipelineNpuBase(text_encoder, model_dir, /*sdxl=*/false, use_v_pred),
        clip_path_(std::move(clip_path)),
        unet_path_(std::move(unet_path)),
        vae_decoder_path_(std::move(vae_decoder_path)),
        vae_encoder_path_(std::move(vae_encoder_path)) {}

  ~PipelineSd15Mtk() override {
    if (clip_session_) clip_interpreter_->releaseSession(clip_session_);
    delete clip_interpreter_;
  }

  bool initialize() override {
    clip_interpreter_ = createMnnInterpreterMmap(clip_path_.c_str());
    if (!clip_interpreter_) {
      QNN_ERROR("Failed load CLIP MNN: %s", clip_path_.c_str());
      return false;
    }
    clip_session_ = createMnnSession(clip_interpreter_, MnnSessionOptions{});
    if (!clip_session_) {
      QNN_ERROR("Failed create persistent MNN CLIP session!");
      return false;
    }
    QNN_INFO("Persistent MNN CLIP session created.");
    auto input =
        clip_interpreter_->getSessionInput(clip_session_, "input_embedding");
    clip_interpreter_->resizeTensor(input, {1, 77, text_embedding_size});
    clip_interpreter_->resizeSession(clip_session_);
    clip_interpreter_->releaseModel();

    auto unet_mtk = mtk_runtime::createModel(unet_path_, "unet");
    if (!unet_mtk) {
      QNN_ERROR("Failed create MTK UNET model.");
      return false;
    }
    auto vae_decoder_mtk = mtk_runtime::createModel(vae_decoder_path_, "vae_decoder");
    if (!vae_decoder_mtk) {
      QNN_ERROR("Failed create MTK VAE Decoder model.");
      return false;
    }
    std::unique_ptr<MtkModel> vae_encoder_mtk;
    if (!vae_encoder_path_.empty()) {
      vae_encoder_mtk = mtk_runtime::createModel(vae_encoder_path_, "vae_encoder");
      if (!vae_encoder_mtk) QNN_WARN("Failed create MTK VAE Enc model.");
    } else {
      QNN_INFO("img2img disabled: VAE encoder not loaded");
    }

    if (mtk_runtime::initializeApp("UNET", unet_mtk) != EXIT_SUCCESS) return false;
    if (mtk_runtime::initializeApp("VAEDecoder", vae_decoder_mtk) != EXIT_SUCCESS)
      return false;
    if (vae_encoder_mtk &&
        mtk_runtime::initializeApp("VAEEncoder", vae_encoder_mtk) != EXIT_SUCCESS)
      return false;

    unet_ = std::move(unet_mtk);
    vae_decoder_ = std::move(vae_decoder_mtk);
    vae_encoder_ = std::move(vae_encoder_mtk);
    return true;
  }

  bool supportsImg2Img() const override { return vae_encoder_ != nullptr; }

 protected:
  bool previewSupported() const override { return true; }
  bool vaeTilingSupported() const override { return false; }

  void encodeText(const ProcessedPromptPair &prompts, bool need_negative,
                  bool need_positive, Conditioning &cond) override {
    if (!clip_interpreter_ || !clip_session_)
      throw std::runtime_error("MNN CLIP missing");

    auto input =
        clip_interpreter_->getSessionInput(clip_session_, "input_embedding");
    clip_interpreter_->resizeTensor(input, {1, 77, text_embedding_size});
    clip_interpreter_->resizeSession(clip_session_);

    auto run_side = [&](const std::vector<float> &embeddings, float *dst) {
      memcpy(input->host<float>(), embeddings.data(),
             77 * text_embedding_size * sizeof(float));
      clip_interpreter_->runSession(clip_session_);
      auto out = clip_interpreter_->getSessionOutput(clip_session_,
                                                     "last_hidden_state");
      memcpy(dst, out->host<float>(), 77 * text_embedding_size * sizeof(float));
    };

    if (need_negative) run_side(prompts.negative_embeddings, cond.negHidden());
    if (need_positive) run_side(prompts.positive_embeddings, cond.posHidden());
  }

  void vaeEncode(const GenerationRequest &, const float *image, float *mean,
                 float *std_dev) override {
    if (!vae_encoder_) throw std::runtime_error("MTK VAE Enc missing");
    if (NpuStatus::SUCCESS != vae_encoder_->executeVaeEncoderGraphs(
                                  const_cast<float *>(image), mean, std_dev))
      throw std::runtime_error("MTK VAE enc exec failed");
  }

  void runUnetStep(const GenerationRequest &, const float *latents_batch2,
                   int timestep, bool skip_uncond, Conditioning &cond,
                   float *out_batch2) override {
    if (!unet_) throw std::runtime_error("MTK UNET missing");

    const int single_latent_size = 1 * 4 * sample_width * sample_height;
    float *latents_in = const_cast<float *>(latents_batch2);

    if (!skip_uncond &&
        NpuStatus::SUCCESS != unet_->executeUnetGraphs(latents_in, timestep,
                                                       cond.negHidden(),
                                                       out_batch2))
      throw std::runtime_error("MTK UNET exec failed (uncond)");

    if (NpuStatus::SUCCESS !=
        unet_->executeUnetGraphs(latents_in + single_latent_size, timestep,
                                 cond.posHidden(),
                                 out_batch2 + single_latent_size))
      throw std::runtime_error("MTK UNET exec failed (cond)");
  }

  void vaeDecode(const GenerationRequest &, const float *latents,
                 float *pixels) override {
    if (!vae_decoder_) throw std::runtime_error("MTK VAE Dec missing");
    if (NpuStatus::SUCCESS != vae_decoder_->executeVaeDecoderGraphs(
                                  const_cast<float *>(latents), pixels))
      throw std::runtime_error("MTK VAE dec exec failed");
  }

 private:
  const std::string clip_path_;
  const std::string unet_path_;
  const std::string vae_decoder_path_;
  const std::string vae_encoder_path_;

  MNN::Interpreter *clip_interpreter_ = nullptr;
  MNN::Session *clip_session_ = nullptr;
};

#endif  // PIPELINESD15MTK_HPP
