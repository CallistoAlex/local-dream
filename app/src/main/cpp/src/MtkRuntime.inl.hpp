#ifndef MTKRUNTIME_INL_HPP
#define MTKRUNTIME_INL_HPP

#include "MtkModel.hpp"

namespace mtk_runtime {

inline std::unique_ptr<MtkModel> createModel(const std::string& modelPath,
                                             const std::string& modelName) {
  if (!g_initialized) {
    QNN_ERROR("MTK runtime not initialized");
    return nullptr;
  }
  return std::make_unique<MtkModel>(modelPath, modelName);
}

inline int initializeApp(const std::string& modelName,
                         std::unique_ptr<MtkModel>& app) {
  if (!app) return EXIT_FAILURE;
  QNN_INFO("Initializing MTK LiteRT model: %s", modelName.c_str());
  if (!app->initialize()) {
    QNN_ERROR("Failed to initialize MTK model: %s", modelName.c_str());
    return EXIT_FAILURE;
  }
  QNN_INFO("MTK LiteRT model initialized: %s", modelName.c_str());
  return EXIT_SUCCESS;
}

inline std::unique_ptr<MtkModel> createAndInitModel(
    const std::string& modelPath, const std::string& modelName) {
  auto app = createModel(modelPath, modelName);
  if (!app) throw std::runtime_error("Failed create MTK model: " + modelName);
  if (initializeApp(modelName, app) != EXIT_SUCCESS)
    throw std::runtime_error("Failed init MTK model: " + modelName);
  return app;
}

}  // namespace mtk_runtime

#endif  // MTKRUNTIME_INL_HPP
