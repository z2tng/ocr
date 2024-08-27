#include "utils/image_utils.h"

namespace utils {

cv::Mat ImageUtils::AdjustImageSize(const cv::Mat &src, int dest_width, int dest_height) {
    float scale = static_cast<float>(dest_width) / src.cols;
    int scaled_width = static_cast<int>(src.cols * scale);

    cv::Mat dest;
    cv::resize(src, dest, cv::Size(scaled_width, dest_height));

    cv::Mat output_image(dest_height, dest_width, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::Rect roi(0, 0, std::min(scaled_width, dest_width), dest_height);
    dest(roi).copyTo(output_image(roi));
    return dest;
}

cv::Mat ImageUtils::GetRotateCropImage(const cv::Mat &src, const std::vector<cv::Point> &points) {
    // 计算 box 的最小外接矩形
    std::vector<int> points_x{points[0].x, points[1].x, points[2].x, points[3].x};
    std::vector<int> points_y{points[0].y, points[1].y, points[2].y, points[3].y};
    int left = int(*std::min_element(points_x.begin(), points_x.end()));
    int right = int(*std::max_element(points_x.begin(), points_x.end()));
    int top = int(*std::min_element(points_y.begin(), points_y.end()));
    int bottom = int(*std::max_element(points_y.begin(), points_y.end()));

    // 裁剪图像
    cv::Mat crop_image = src(cv::Rect(left, top, right - left, bottom - top)).clone();

    // 调整 box 点的坐标
    std::vector<cv::Point2f> adjust_points{
        cv::Point2f(points[0].x - left, points[0].y - top),
        cv::Point2f(points[1].x - left, points[1].y - top),
        cv::Point2f(points[2].x - left, points[2].y - top),
        cv::Point2f(points[3].x - left, points[3].y - top)
    };

    // 计算裁剪后的图像的宽高
    int crop_width = cv::norm(adjust_points[0] - adjust_points[1]);
    int crop_height = cv::norm(adjust_points[0] - adjust_points[3]);

    // 计算透视变换后的矩形区域
    std::vector<cv::Point2f> dest_points = {
        cv::Point2f(0.f, 0.f),
        cv::Point2f(crop_width, 0.f),
        cv::Point2f(crop_width, crop_height),
        cv::Point2f(0.f, crop_height)
    };

    // 透视变换
    cv::Mat transform_mat = cv::getPerspectiveTransform(adjust_points, dest_points);
    cv::Mat rotate_crop_image;
    cv::warpPerspective(crop_image, rotate_crop_image, transform_mat, cv::Size(crop_width, crop_height));

    // 如果高度超过宽度 1.5 倍，则旋转 90 度
    if (static_cast<float>(crop_height) >= crop_width * 1.5f) {
        cv::rotate(rotate_crop_image, rotate_crop_image, cv::ROTATE_90_CLOCKWISE);
    }
    return rotate_crop_image;
}

int ImageUtils::GetThickness(const cv::Mat &box_image) {
    int min_side = std::min(box_image.cols, box_image.rows);
    int thickness = min_side / 1000 + 2;
    return thickness;
}

base::ScaleParam ImageUtils::GetScaleParam(const cv::Mat &src, float scale) {
    int src_width = src.cols;
    int src_height = src.rows;
    int dest_width = static_cast<int>(src_width * scale);
    int dest_height = static_cast<int>(src_height * scale);

    dest_width = dest_width % 32 == 0 ? dest_width : (dest_width / 32 + 1) * 32;
    dest_height = dest_height % 32 == 0 ? dest_height : (dest_height / 32 + 1) * 32;

    float scale_w = static_cast<float>(dest_width) / src_width;
    float scale_h = static_cast<float>(dest_height) / src_height;
    return {src_width, src_height, dest_width, dest_height, scale_w, scale_h};
}

base::ScaleParam ImageUtils::GetScaleParam(const cv::Mat &src, int target_max_side_len) {
    int max_side = std::max(src.cols, src.rows);
    float scale = target_max_side_len <= 0 || target_max_side_len >= max_side ? 1.0f : static_cast<float>(target_max_side_len) / max_side;
    return GetScaleParam(src, scale);
}

} // namespace utils
