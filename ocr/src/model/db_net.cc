#include "model/db_net.h"
#include "utils/ocr_utils.h"
#include "utils/time_utils.h"

#include <numeric>

namespace model {

DbNet::DbNet()
        : num_threads_(0),
          env_(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DbNet")),
          session_options_(),
          input_name_(),
          output_name_() {}

DbNet::~DbNet() {}

void DbNet::SetNumThreads(int num_threads) {
    num_threads_ = num_threads;
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

void DbNet::Init(const std::string &model_path) {
    session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(), session_options_);

    utils::OcrUtils::GetInputName(session_, input_name_);
    utils::OcrUtils::GetOutputName(session_, output_name_);
}

std::vector<base::TextBox> DbNet::FindRsBoxes(const cv::Mat &feat, const cv::Mat &binary_feat, base::ScaleParam &scale_param, float box_score_threshold, float box_threshold, float unclip_ratio) {
    const float min_area = 3.0;
    std::vector<base::TextBox> boxes;
    std::vector<std::vector<cv::Point>> contours;
    findContours(binary_feat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (const auto &contour : contours) {
        // 计算最小外接矩形的四个顶点
        float min_side_len, perimeter;
        std::vector<cv::Point> min_box = utils::OcrUtils::GetMinBoxes(contour, min_side_len, perimeter);
        if (min_side_len < min_area) continue;

        // 计算box得分
        float score = utils::OcrUtils::BoxScoreFast(feat, contour);
        if (score < box_score_threshold) continue;

        // 截取 box
        auto clip_box = utils::OcrUtils::UnClip(min_box, perimeter, unclip_ratio);
        auto clip_min_box = utils::OcrUtils::GetMinBoxes(clip_box, min_side_len, perimeter);
        if (min_side_len < min_area + 2) continue;

        for (auto &point : clip_min_box) {
            point.x = point.x / scale_param.ratio_w;
            point.x = std::min(std::max(0, point.x), scale_param.src_width);
            point.y = point.y / scale_param.ratio_h;
            point.y = std::min(std::max(0, point.y), scale_param.src_height);
        }
        boxes.emplace_back(base::TextBox{clip_min_box, score});
    }
    reverse(boxes.begin(), boxes.end());
    return boxes;
}

std::vector<base::TextBox> DbNet::GetTextBoxes(cv::Mat &src, base::ScaleParam &scale_param, float box_score_threshold, float box_threshold, float unclip_ratio) {
    cv::Mat src_resize;
    cv::resize(src, src_resize, cv::Size(scale_param.dest_width, scale_param.dest_height));

    // 预处理
    auto input_data = utils::OcrUtils::SubstractMeanNormalize(src_resize, mean_, norm_);
    std::array<int64_t, 4> input_shape{1, src_resize.channels(), src_resize.rows, src_resize.cols};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 创建输入tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names = {input_name_.c_str()};
    std::vector<const char*> output_names = {output_name_.c_str()};

    // 获取输出tensor
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());

    // 获取输出tensor的数据
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    // 获取特征图
    cv::Mat feat(src_resize.rows, src_resize.cols, CV_32FC1, output_data);
    cv::Mat binary_feat = feat > box_threshold;

    // 查找文本框
    return FindRsBoxes(feat, binary_feat, scale_param, box_score_threshold, box_threshold, unclip_ratio);
}

} // namespace model