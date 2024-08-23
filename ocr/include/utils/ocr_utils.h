#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

#include "base/ocr_structs.h"

namespace utils {

class OcrUtils {
public:
    static void GetInputName(std::shared_ptr<Ort::Session> session, std::string &input_name);
    static void GetOutputName(std::shared_ptr<Ort::Session> session, std::string &output_name);

    static std::vector<float> SubstractMeanNormalize(const cv::Mat &image, const std::vector<float> &mean, const std::vector<float> &norm);

    static std::vector<cv::Point> GetMinBoxes(const std::vector<cv::Point> &points, float &min_side_len, float &perimeter);
    static float BoxScoreFast(const cv::Mat &feat, const std::vector<cv::Point> &box);
    static std::vector<cv::Point> UnClip(const std::vector<cv::Point> &box, float perimeter, float unclip_ratio);

    static std::vector<int> GetAngleIndexes(const std::vector<base::Angle> &angles);

    static void DrawTextBox(cv::Mat &src, const cv::RotatedRect &rect, int thickness);
    static void DrawTextBox(cv::Mat &src, const std::vector<cv::Point> &points, int thickness);
    static void DrawTextBoxes(cv::Mat &src, const std::vector<base::TextBox> &boxes, int thickness);
};

} // namespace utils

