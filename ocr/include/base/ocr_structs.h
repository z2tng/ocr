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

} // namespace base