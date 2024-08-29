#pragma once

#include "base/ocr_structs.h"

#include <onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>

#include <memory>
#include <string>

namespace model {

class DbNet {
public:
    DbNet();
    ~DbNet();

    void Init(const std::string &model_path);
    void SetNumThreads(int num_threads);

    std::vector<base::TextBox> GetTextBoxes(cv::Mat &src, base::ScaleParam &scale_param, float box_score_threshold, float box_threshold, float unclip_ratio);

private:
    std::vector<base::TextBox> FindRsBoxes(const cv::Mat &feat, const cv::Mat &binary_feat, base::ScaleParam &scale_param, float box_score_threshold, float unclip_ratio);

    int num_threads_;

    std::shared_ptr<Ort::Session> session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;

    std::string input_name_;
    std::string output_name_;

    const std::vector<float> mean_{0.485 * 255, 0.456 * 255, 0.406 * 255};
    const std::vector<float> norm_{1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0};
};

} // namespace model