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
    cv::RotatedRect rect = cv::minAreaRect(points);

    // 计算最小外接矩形的四个顶点
    cv::Mat box_points;
    cv::boxPoints(rect, box_points);

    // 将box_points转换为float类型的指针
    float *p1 = reinterpret_cast<float*>(box_points.data);
    std::vector<cv::Point> temp_box;
    for (int i = 0; i < 4; i++) {
        temp_box.emplace_back(cv::Point2f(p1[i * 2], p1[i * 2 + 1]));
    }

    // 对temp_box按照x坐标进行排序
    std::sort(temp_box.begin(), temp_box.end(), [](const cv::Point &a, const cv::Point &b) {
        return a.x < b.x;
    });

    // 根据y坐标的大小，得到四个点的索引
    int index1, index2, index3, index4;
    if (temp_box[0].y < temp_box[1].y) {
        index1 = 0;
        index4 = 1;
    } else {
        index1 = 1;
        index4 = 0;
    }

    if (temp_box[2].y < temp_box[3].y) {
        index2 = 2;
        index3 = 3;
    } else {
        index2 = 3;
        index3 = 2;
    }

    min_box.clear();
    min_box.emplace_back(temp_box[index1]);
    min_box.emplace_back(temp_box[index2]);
    min_box.emplace_back(temp_box[index3]);
    min_box.emplace_back(temp_box[index4]);

    min_side_len = std::min(rect.size.width, rect.size.height);
    perimeter = 2.0f * (rect.size.width + rect.size.height);

    return min_box;
}

float OcrUtils::BoxScoreFast(const cv::Mat &feat, const std::vector<cv::Point> &box) {
    int width = feat.cols;
    int height = feat.rows;

    // 计算边界框
    cv::Rect bounding_box = cv::boundingRect(box);
    int min_x = std::max(0, bounding_box.x);
    int max_x = std::min(width - 1, bounding_box.x + bounding_box.width);
    int min_y = std::max(0, bounding_box.y);
    int max_y = std::min(height - 1, bounding_box.y + bounding_box.height);

    // 创建 mask
    cv::Mat mask(max_y - min_y + 1, max_x - min_x + 1, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::fillPoly(mask, {box}, cv::Scalar(1, 1, 1), 1);

    // 计算指定区域的平均值，即平均像素强度
    cv::Mat region(feat(cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y)).clone());
    return cv::mean(region, mask).val[0];
}

std::vector<cv::Point> OcrUtils::UnClip(const std::vector<cv::Point> &box, float perimeter, float unclip_ratio) {
    ClipperLib::Path poly;
    for (const auto &point : box) {
        poly.push_back(ClipperLib::IntPoint(point.x, point.y));
    }

    double distance = unclip_ratio * ClipperLib::Area(poly) / static_cast<double>(perimeter);

    ClipperLib::ClipperOffset offset;
    offset.AddPath(poly, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths polys;
    polys.push_back(poly);
    offset.Execute(polys, distance);

    std::vector<cv::Point> out_box;
    for (const auto &path : polys) {
        for (const auto &point : path) {
            out_box.emplace_back(cv::Point(point.X, point.Y));
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
