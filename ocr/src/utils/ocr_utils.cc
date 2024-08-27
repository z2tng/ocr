#include "utils/ocr_utils.h"
#include "utils/clipper.hpp"

#include <onnxruntime_cxx_api.h>

namespace utils {

void OcrUtils::GetInputName(std::shared_ptr<Ort::Session> session, std::string &input_name) {
    size_t num_input_nodes = session->GetInputCount();
    if (num_input_nodes > 0) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char* in = session->GetInputName(0, allocator);
            input_name = std::string(in);
            allocator.Free(in);
        }
    }
}

void OcrUtils::GetOutputName(std::shared_ptr<Ort::Session> session, std::string &output_name) {
    size_t num_output_nodes = session->GetOutputCount();
    if (num_output_nodes > 0) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char* out = session->GetOutputName(0, allocator);
            output_name = std::string(out);
            allocator.Free(out);
        }
    }
}

std::vector<float> OcrUtils::SubstractMeanNormalize(const cv::Mat &image, const std::vector<float> &mean, const std::vector<float> &norm) {
    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC3);

    cv::Mat normalized_image;
    cv::subtract(float_image, cv::Scalar(mean[0], mean[1], mean[2]), normalized_image);
    cv::multiply(normalized_image, cv::Scalar(norm[0], norm[1], norm[2]), normalized_image);

    std::vector<float> input_data;
    input_data.assign((float*)normalized_image.datastart, (float*)normalized_image.dataend);
    return input_data;
}

std::vector<cv::Point> OcrUtils::GetMinBoxes(const std::vector<cv::Point> &points, float &min_side_len, float &perimeter) {
    std::vector<cv::Point> min_box;
    // 输出point的数据类型，是否是浮点型
    cv::RotatedRect rect = cv::minAreaRect(points);

    // 计算最小外接矩形的四个顶点
    cv::Mat box_points;
    cv::boxPoints(rect, box_points);
    std::vector<cv::Point> temp_box(4);
    for (int i = 0; i < 4; i++) {
        temp_box[i] = box_points.at<cv::Point2f>(i);
    }
    // 对temp_box按照x坐标进行排序
    std::sort(temp_box.begin(), temp_box.end(), [](const cv::Point &a, const cv::Point &b) {
        return a.x < b.x;
    });

    // 根据y坐标的大小，得到四个点的索引
    int left_top = 0, left_bottom = 1;
    int right_top = 2, right_bottom = 3;
    if (temp_box[0].y > temp_box[1].y) {
        left_top = 1;
        left_bottom = 0;
    }
    if (temp_box[2].y > temp_box[3].y) {
        right_top = 3;
        right_bottom = 2;
    }
    min_box.emplace_back(temp_box[left_top]);
    min_box.emplace_back(temp_box[right_top]);
    min_box.emplace_back(temp_box[right_bottom]);
    min_box.emplace_back(temp_box[left_bottom]);

    min_side_len = std::min(rect.size.width, rect.size.height);
    perimeter = 2.0f * (rect.size.width + rect.size.height);

    return min_box;
}

float OcrUtils::BoxScoreFast(const cv::Mat &feat, const std::vector<cv::Point> &box) {
    std::vector<cv::Point> box_temp(box);
    int width = feat.cols;
    int height = feat.rows;

    int min_x = width - 1, min_y = height - 1;
    int max_x = 0, max_y = 0;
    // 计算最小外接矩形的两个顶点
    for (const auto &point : box) {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }

    for (auto &point : box_temp) {
        point.x -= min_x;
        point.y -= min_y;
    }

    // 创建 mask
    cv::Mat mask(max_y - min_y + 1, max_x - min_x + 1, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::fillPoly(mask, {box_temp}, cv::Scalar(1, 1, 1), 1);

    // 计算指定区域的平均值，即平均像素强度
    cv::Mat region(feat(cv::Rect(cv::Point(min_x, min_y), cv::Point(max_x + 1, max_y + 1))).clone());
    return cv::mean(region, mask).val[0];
}

std::vector<cv::Point> OcrUtils::UnClip(const std::vector<cv::Point> &box, float perimeter, float unclip_ratio) {
    ClipperLib::Path poly;
    for (const auto &point : box) {
        poly.push_back(ClipperLib::IntPoint(point.x, point.y));
    }

    double distance = unclip_ratio * ClipperLib::Area(poly) / static_cast<double>(perimeter);

    ClipperLib::ClipperOffset offset;
    offset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);

    ClipperLib::Paths polys;
    polys.push_back(poly);
    offset.Execute(polys, distance);

    std::vector<cv::Point> out_box;
    for (const auto &poly : polys) {
        for (const auto &point : poly) {
            out_box.emplace_back(point.X, point.Y);
        }
    }
    return out_box;
}

std::vector<int> OcrUtils::GetAngleIndexes(const std::vector<base::Angle> &angles) {
    std::vector<int> result(angles.size());
    for (size_t i = 0; i < angles.size(); i++) {
        result[i] = angles[i].index;
    }
    return result;
}

void OcrUtils::DrawTextBox(cv::Mat &src, const cv::RotatedRect &rect, int thickness) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++) {
        cv::line(src, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), thickness);
    }
}

void OcrUtils::DrawTextBox(cv::Mat &src, const std::vector<cv::Point> &points, int thickness) {
    auto color = cv::Scalar(0, 0, 255);
    for (int i = 0; i < 4; i++) {
        cv::line(src, points[i], points[(i + 1) % 4], color, thickness);
    }
}

void OcrUtils::DrawTextBoxes(cv::Mat &src, const std::vector<base::TextBox> &boxes, int thickness) {
    for (const auto &box : boxes) {
        DrawTextBox(src, box.points, thickness);
    }
}

} // namespace utils
