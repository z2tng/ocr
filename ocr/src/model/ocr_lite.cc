#include "model/ocr_lite.h"
#include "utils/ocr_utils.h"
#include "utils/file_utils.h"
#include "utils/image_utils.h"
#include "utils/time_utils.h"

namespace model {

void OcrLite::Init(const std::string &det_path, const std::string &cls_path, const std::string &rec_path, const std::string &keys_path) {
    angle_net_.Init(cls_path);
    db_net_.Init(det_path);
    crnn_net_.Init(rec_path, keys_path);
}

void OcrLite::SetNumThreads(int num_threads) {
    angle_net_.SetNumThreads(num_threads);
    db_net_.SetNumThreads(num_threads);
    crnn_net_.SetNumThreads(num_threads);
}

base::OcrResult OcrLite::Process(const std::string &path, const std::string &image_name, int padding, int max_side_len, float box_score_threshold, float box_threshold, float unclip_ratio, bool cal_angle, bool cal_most_angle) {
    std::string image_path = utils::FileUtils::JoinPath(path, image_name);

    cv::Mat src_bgr = cv::imread(path, cv::IMREAD_COLOR); // BGR image
    cv::Mat src_rgb;
    cv::cvtColor(src_bgr, src_rgb, cv::COLOR_BGR2RGB); // RGB image

    // 
    int max_side = std::max(src_rgb.cols, src_rgb.rows);
    int resize = max_side_len <= 0 || max_side_len >= max_side ? max_side : max_side_len;
    resize += 2 * padding;
    cv::Rect padding_rect(padding, padding, src_rgb.cols, src_rgb.rows);
    cv::Mat src_padding = MakePadding(src_rgb, padding);
    base::ScaleParam scale_param = utils::ImageUtils::GetScaleParam(src_padding, resize);

    return process(path, image_name, src_padding, padding_rect, scale_param, box_score_threshold, box_threshold, unclip_ratio, cal_angle, cal_most_angle);
}

base::OcrResult OcrLite::Process(cv::Mat &src, int padding, int max_side_len, float box_score_threshold, float box_threshold, float unclip_ratio, bool cal_angle, bool cal_most_angle) {
    int max_side = std::max(src.cols, src.rows);
    int resize = max_side_len <= 0 || max_side_len >= max_side ? max_side : max_side_len;
    resize += 2 * padding;
    cv::Rect padding_rect(padding, padding, src.cols, src.rows);
    cv::Mat src_padding = MakePadding(src, padding);
    base::ScaleParam scale_param = utils::ImageUtils::GetScaleParam(src_padding, resize);

    std::string image_name = "image" + std::to_string(utils::TimeUtils::now());
    return process(output_path_, image_name, src, padding_rect, scale_param, box_score_threshold, box_threshold, unclip_ratio, cal_angle, cal_most_angle);
}

cv::Mat OcrLite::MakePadding(cv::Mat &src, const int padding, const cv::Scalar &padding_value) {
    if (padding <= 0) return src;

    cv::Mat result;
    if (src.empty()) return result;

    cv::copyMakeBorder(src, result, padding, padding, padding, padding, cv::BORDER_CONSTANT, padding_value);
    return result;
}

std::vector<cv::Mat> OcrLite::GetBoxImages(const cv::Mat &src, std::vector<base::TextBox> &boxes, const std::string &path, const std::string &image_name) {
    std::vector<cv::Mat> box_images;
    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::Mat box_image = utils::ImageUtils::GetRotateCropImage(src, boxes[i].points);
        box_images.emplace_back(box_image);

        if (is_output_part_image_) {
            std::string image_path = utils::FileUtils::JoinPath(path, image_name + "_" + std::to_string(i) + ".jpg");
            cv::imwrite(image_path, box_image);
        }
    }
    return box_images;
}

base::OcrResult OcrLite::process(const std::string &path, const std::string &image_name, cv::Mat &src, cv::Rect &orignal_rect, base::ScaleParam &scale_param, float box_score_threshold, float box_threshold, float unclip_ratio, bool cal_angle, bool cal_most_angle) {
    cv::Mat text_box_padding_image = src.clone();
    int thickness = utils::ImageUtils::GetThickness(src);

    // 文本检测
    double det_start = utils::TimeUtils::now();
    std::vector<base::TextBox> boxes = db_net_.GetTextBoxes(src, scale_param, box_score_threshold, box_threshold, unclip_ratio);
    double det_end = utils::TimeUtils::now();
    double det_time = det_end - det_start;
    // TODO: LOG_INFO det

    utils::OcrUtils::DrawTextBoxes(text_box_padding_image, boxes, thickness);

    // 角度检测
    std::vector<cv::Mat> box_images = GetBoxImages(src, boxes, path, image_name);
    std::vector<base::Angle> angles = angle_net_.GetAngles(box_images, path, image_name, cal_angle, cal_most_angle);
    // TODO: LOG_INFO cls

    // 文本识别
    // 根据角度旋转文本框
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (angles[i].index == 0) {
            cv::rotate(box_images[i], box_images[i], cv::ROTATE_180);
        }
    }
    
    std::vector<base::TextLine> text_lines = crnn_net_.GetTextLines(box_images, path, image_name);
    // TODO: LOG_INFO rec

    // 合并结果
    std::vector<base::TextBlock> text_blocks;
    for (size_t i = 0; i < text_lines.size(); ++i) {
        std::vector<cv::Point> box_points = {
            boxes[i].points[0] - cv::Point(orignal_rect.x, orignal_rect.y),
            boxes[i].points[1] - cv::Point(orignal_rect.x, orignal_rect.y),
            boxes[i].points[2] - cv::Point(orignal_rect.x, orignal_rect.y),
            boxes[i].points[3] - cv::Point(orignal_rect.x, orignal_rect.y)
        };
        text_blocks.emplace_back(
            base::TextBlock{
                box_points,
                boxes[i].score,
                angles[i].index,
                angles[i].score,
                angles[i].time,
                text_lines[i].text,
                text_lines[i].char_scores,
                text_lines[i].time,
                angles[i].time + text_lines[i].time
            }
        );
    }

    double full_time = utils::TimeUtils::now() - det_start;

    // 修剪图片至原始大小，并转换为BGR格式
    cv::Mat rgb_box_image, text_box_image;
    if (orignal_rect.x > 0 && orignal_rect.y > 0) {
        rgb_box_image = text_box_padding_image(orignal_rect).clone();
    } else {
        rgb_box_image = text_box_padding_image;
    }
    cv::cvtColor(rgb_box_image, text_box_image, cv::COLOR_RGB2BGR);

    // 保存结果
    if (is_output_result_image_) {
        std::string image_path = utils::FileUtils::JoinPath(output_path_, image_name + "_result.jpg");
        cv::imwrite(image_path, text_box_image);
    }
    
    std::string str_result;
    for (const auto &block : text_blocks) {
        str_result += block.text + "\n";
    }

    return base::OcrResult{
        text_blocks,
        text_box_image,
        full_time,
        str_result,
        det_time
    };
}
    
} // namespace name