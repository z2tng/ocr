#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

namespace base {

struct ScaleParam {
    int src_width;
    int src_height;
    int dest_width;
    int dest_height;
    float ratio_w;
    float ratio_h;
};

struct TextBox {
    std::vector<cv::Point> points;
    float score;
};

struct Angle {
    int index;
    float score;
    double time;
};

struct TextLine {
    std::string text;
    std::vector<float> char_scores;
    double time;
};

struct TextBlock {
    // 文本框检测结果
    std::vector<cv::Point> box_points;
    float box_score;
    // 角度检测结果
    int angle_index;
    float angle_score;
    double angle_time;
    // 文本识别结果
    std::string text;
    std::vector<float> char_scores;
    double crnn_time;

    double block_time;
};

struct OcrResult {
    std::vector<TextBlock> blocks;
    cv::Mat box_image;
    double box_time;

    std::string str_result;
    double det_time;
};

} // namespace base