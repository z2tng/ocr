#include "model/angle_net.h"
#include "utils/ocr_utils.h"
#include "utils/image_utils.h"
#include "utils/time_utils.h"

#include <array>
#include <numeric>

namespace model {

AngleNet::AngleNet()
        : is_output_debug_image_(false),
          num_threads_(0),
          env_(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "AngleNet")),
          session_options_(),
          input_name_(),
          output_name_() {}

AngleNet::~AngleNet() {}

void AngleNet::SetNumThreads(int num_threads) {
    num_threads_ = num_threads;
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

void AngleNet::Init(const std::string &model_path) {
    session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(), session_options_);

    utils::OcrUtils::GetInputName(session_, input_name_);
    utils::OcrUtils::GetOutputName(session_, output_name_);
}

std::vector<base::Angle> AngleNet::GetAngles(const std::vector<cv::Mat> &images, const std::string &path, const std::string &image_name, bool cal_angle, bool cal_most_angle) {
    int size = images.size();
    std::vector<base::Angle> angles(size);

    if (cal_angle) {
        for (int i = 0; i < size; i++) {
            double start_time = utils::TimeUtils::now();
            auto image = utils::ImageUtils::AdjustImageSize(images[i], dest_width_, dest_height_);
            angles[i] = run(image);
            double end_time = utils::TimeUtils::now();
            angles[i].time = end_time - start_time;
            // LOG(INFO) << "AngleNet time: " << angles[i].time;

            if (is_output_debug_image_) {
                std::string angle_image_path = path + "/" + image_name + "_angle_" + std::to_string(i) + ".jpg";
                cv::Mat angle_image = images[i].clone();
                cv::imwrite(angle_image_path, angle_image);
            }
        }

        if (cal_most_angle) {
            std::vector<int> indexes = utils::OcrUtils::GetAngleIndexes(angles);
            double sum = std::accumulate(indexes.begin(), indexes.end(), 0.0);
            double half = size / 2;

            int most_index = sum < half ? 0 : 1;
            for (int i = 0; i < size; i++) {
                angles[i].index = indexes[most_index];
            }
        }
    } else {
        for (int i = 0; i < size; i++) {
            angles[i] = base::Angle{-1, 0.0f};
        }
    }

    return angles;
}

base::Angle AngleNet::run(const cv::Mat &src) {
    std::vector<float> input_tensor_values = utils::OcrUtils::SubstractMeanNormalize(src, mean_, norm_);

    std::array<int64_t, 4> input_shape{1, 3, src.rows, src.cols};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());
    assert(input_tensor.IsTensor());

    std::vector<const char *> input_names = {input_name_.c_str()};
    std::vector<const char *> output_names = {output_name_.c_str()};
    auto output_tensor = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());

    std::vector<int64_t> output_shape(output_tensor[0].GetTensorTypeAndShapeInfo().GetShape());
    int64_t output_count = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());

    float *output = output_tensor.front().GetTensorMutableData<float>();
    std::vector<float> output_values(output, output + output_count);
    
    return ScoreToAngle(output_values);
}

base::Angle AngleNet::ScoreToAngle(const std::vector<float> &output_values) {
    int max_index = 0;
    float max_value = output_values.empty() ? -1000.0f : output_values[0];

    for (size_t i = 0; i < output_values.size(); i++) {
        if (output_values[i] > max_value) {
            max_value = output_values[i];
            max_index = i;
        }
    }
    return base::Angle{max_index, max_value};
}

} // namespace model