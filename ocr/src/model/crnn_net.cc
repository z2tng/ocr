#include "model/crnn_net.h"
#include "utils/ocr_utils.h"
#include "utils/time_utils.h"

#include <fstream>
#include <numeric>

namespace model {

CrnnNet::CrnnNet()
        : is_output_debug_image_(false),
          num_threads_(0),
          env_(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "CrnnNet")),
          session_options_(),
          input_name_(),
          output_name_() {}

CrnnNet::~CrnnNet() {}

void CrnnNet::Init(const std::string &model_path, const std::string &keys_path) {
    session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(), session_options_);

    utils::OcrUtils::GetInputName(session_, input_name_);
    utils::OcrUtils::GetOutputName(session_, output_name_);

    std::ifstream infile(keys_path);
    std::string line;
    if (infile) {
        while (getline(infile, line)) {
            keys_.push_back(line);
        }
    } else {
        // LOG_FATAL << "Failed to open keys file: " << keys_path;
    }

    if (keys_.size() != 5531) {
        // LOG_ERROR << "Missing keys";
    }
    // LOG_INFO << "Keys size: " << keys_.size();
}

void CrnnNet::SetNumThreads(int num_threads) {
    num_threads_ = num_threads;
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

base::TextLine CrnnNet::ScoreToTextLine(const std::vector<float> &output_values, int h, int w) {
    // 将输出的分数转换为文本行
    int size = keys_.size();
    std::string str_result;
    std::vector<float> scores;
    int last_index = -1;

    std::vector<float> exp_values(w);

    // 逐行计算最大值
    for (int i = 0; i < h; ++i) {
        // Softmax 计算
        float sum = 0.0f;
        int max_index = 0;
        float max_value = -1000.0f;

        for (int j = 0; j < w; ++j) {
            exp_values[j] = exp(output_values[i * w + j]);
            sum += exp_values[j];
            if (exp_values[j] > max_value) {
                max_value = exp_values[j];
                max_index = j;
            }
        }
        max_value /= sum; // Softmax 归一化

        // 过滤掉相邻重复的字符
        if (max_index > 0 && max_index < size && max_index != last_index) {
            str_result += keys_[max_index];
            scores.push_back(max_value);
        }
        last_index = max_index;
    }
    return {str_result, scores};
}

base::TextLine CrnnNet::run(const cv::Mat &src) {
    float scale = static_cast<float>(dest_height_) / src.rows;
    int dest_width = static_cast<int>(src.cols * scale);

    cv::Mat src_resize;
    cv::resize(src, src_resize, cv::Size(dest_width, dest_height_));

    std::vector<float> input_data = utils::OcrUtils::SubstractMeanNormalize(src_resize, mean_, norm_);
    std::vector<int64_t> input_shape = {1, src_resize.channels(), src_resize.rows, src_resize.cols};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
    
    std::vector<const char*> input_names = {input_name_.c_str()};
    std::vector<const char*> output_names = {output_name_.c_str()};
    std::vector<Ort::Value> output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());

    std::vector<int64_t> output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t output_count = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());

    float* output = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> output_values(output, output + output_count);
    return ScoreToTextLine(output_values, output_shape[0], output_shape[2]);
}

std::vector<base::TextLine> CrnnNet::GetTextLines(const std::vector<cv::Mat> &images, const std::string &path, const std::string &image_name) {
    int size = images.size();
    std::vector<base::TextLine> text_lines(size);

    for (int i = 0; i < size; ++i) {
        if (is_output_debug_image_) {
            std::string text_image_path = path + "/" + image_name + "_text_" + std::to_string(i) + ".jpg";
            cv::Mat text_image = images[i].clone();
            cv::imwrite(text_image_path, text_image);
        }

        double start = utils::TimeUtils::now();
        text_lines[i] = run(images[i]);
        double end = utils::TimeUtils::now();
        text_lines[i].time = end - start;
    }
    return text_lines;
}

} // namespace model