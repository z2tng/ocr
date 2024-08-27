#pragma once

#include "base/ocr_structs.h"
#include "model/angle_net.h"
#include "model/db_net.h"
#include "model/crnn_net.h"

#include <opencv4/opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <string>

namespace model {

class OcrLite {
public:
    OcrLite() : is_output_console_(false),
                is_output_part_image_(false),
                is_output_result_text_(false),
                is_output_result_image_(false),
                output_path_("./") {}
    ~OcrLite() = default;

    void SetOutputConsole(bool is_output_console) { is_output_console_ = is_output_console; }
    void SetOutputPartImage(bool is_output_part_image) { is_output_part_image_ = is_output_part_image; }
    void SetOutputResultText(bool is_output_result_text) { is_output_result_text_ = is_output_result_text; }
    void SetOutputResultImage(bool is_output_result_image) { is_output_result_image_ = is_output_result_image; }

    void SetOutputPath(const std::string &output_path) { output_path_ = output_path; }

    void Init(const std::string &det_path, const std::string &cls_path, const std::string &rec_path, const std::string &keys_path);
    void SetNumThreads(int num_threads);

    base::OcrResult Process(const std::string &image_dir, const std::string &image_name, int padding, int max_side_len, float box_score_threshold, float box_threshold, float unclip_ratio, bool cal_angle, bool cal_most_angle);

    base::OcrResult Process(cv::Mat &src, int padding, int max_side_len, float box_score_threshold, float box_threshold, float unclip_ratio, bool cal_angle, bool cal_most_angle);

private:
    cv::Mat MakePadding(cv::Mat &src, const int padding, const cv::Scalar &padding_value = {255, 255, 255});

    std::vector<cv::Mat> GetBoxImages(const cv::Mat &src, std::vector<base::TextBox> &boxes, const std::string &path, const std::string &image_name);

    base::OcrResult process(const std::string &path, const std::string &image_name,cv::Mat &src, cv::Rect &orignal_rect, base::ScaleParam &scale_param, float box_score_threshold = 0.6f, float box_threshold = 0.3f, float unclip_ratio = 2.0f, bool cal_angle = true, bool cal_most_angle = true);

    bool is_output_console_;
    bool is_output_part_image_;
    bool is_output_result_text_;
    bool is_output_result_image_;

    std::string output_path_; // 默认为pwd

    AngleNet angle_net_;
    DbNet db_net_;
    CrnnNet crnn_net_;
};
    
} // namespace model
