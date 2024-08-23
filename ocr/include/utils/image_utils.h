#pragma once

#include "base/ocr_structs.h"

#include <opencv4/opencv2/opencv.hpp>

namespace utils {

class ImageUtils {
public:
    static cv::Mat AdjustImageSize(const cv::Mat &src, int dest_width, int dest_height);
};

}
