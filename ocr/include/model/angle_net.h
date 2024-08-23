#pragma once

#include "base/ocr_structs.h"

#include <onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>

#include <vector>
#include <memory>

namespace model {

class AngleNet {
public:
    AngleNet();
    ~AngleNet();

    void Init(const std::string &model_path);
    void SetNumThreads(int num_threads);

    std::vector<base::Angle> GetAngles(const std::vector<cv::Mat> &images, const std::string &path, const std::string &image_name, bool cal_angle, bool cal_most_angle);

private:
    base::Angle run(const cv::Mat &image);
    base::Angle ScoreToAngle(const std::vector<float> &output_values);

    bool is_output_debug_image_;
    int num_threads_;

    std::shared_ptr<Ort::Session> session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;

    std::string input_name_;
    std::string output_name_;

    const std::vector<float> mean_{127.5, 127.5, 127.5};
    const std::vector<float> norm_{1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};

    const int dest_width_ = 192;
    const int dest_height_ = 64;
};

} // namespace model