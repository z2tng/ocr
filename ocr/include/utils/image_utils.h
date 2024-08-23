#pragma once

#include "base/ocr_structs.h"

#include <opencv4/opencv2/opencv.hpp>

#include <vector>

namespace utils {

class ImageUtils {
public:
    static cv::Mat AdjustImageSize(const cv::Mat &src, int dest_width, int dest_height);

    static cv::Mat GetRotateCropImage(const cv::Mat &src, const std::vector<cv::Point> &points);

    static int GetThickness(const cv::Mat &box_image);

    static base::ScaleParam GetScaleParam(const cv::Mat &src, float scale);
    static base::ScaleParam GetScaleParam(const cv::Mat &src, int target_max_side_len);
};

}
