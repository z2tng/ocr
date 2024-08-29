#include "model/ocr_lite.h"
#include "utils/ocr_utils.h"
#include "utils/file_utils.h"

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

void PrintUsage() {
    std::cout << "Usage: ocr_lite [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --models_dir <path>       Path to the directory containing the OCR models" << std::endl;
    std::cout << "  --det_path <path>         Path to the detection model" << std::endl;
    std::cout << "  --cls_path <path>         Path to the classification model" << std::endl;
    std::cout << "  --rec_path <path>         Path to the recognition model" << std::endl;
    std::cout << "  --keys_path <path>        Path to the keys file" << std::endl;
    std::cout << "  --image_path <path>       Path to the image file or directory" << std::endl;
    std::cout << "  --num_threads <int>       Number of threads to use" << std::endl;
    std::cout << "  --padding <int>           Padding size" << std::endl;
    std::cout << "  --max_side_len <int>      Maximum side length" << std::endl;
    std::cout << "  --box_score_threshold <float>  Box score threshold" << std::endl;
    std::cout << "  --box_threshold <float>   Box threshold" << std::endl;
    std::cout << "  --unclip_ratio <float>    Unclip ratio" << std::endl;
    std::cout << "  --cal_angle <bool>        Whether to calculate the angle" << std::endl;
    std::cout << "  --cal_most_angle <bool>   Whether to calculate the most angle" << std::endl;
}

void GetOpt(std::unordered_map<std::string, std::string> &opt_map, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        std::string opt = argv[i];
        if (opt[0] == '-') {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opt_map[opt] = argv[i + 1];
                i++;
            } else {
                opt_map[opt] = "";
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        PrintUsage();
        return -1;
    }

    // Get options
    std::string models_dir;
    std::string det_path, cls_path, rec_path, keys_path;
    std::string image_path, image_dir;
    int num_threads = 4;
    int padding = 50;
    int max_side_len = 1024;
    float box_score_threshold = 0.6f;
    float box_threshold = 0.3f;
    float unclip_ratio = 2.0f;
    bool cal_angle = true;
    bool cal_most_angle = true;

    std::unordered_map<std::string, std::string> opt_map;
    GetOpt(opt_map, argc, argv);

    for (auto &opt : opt_map) {
        if (opt.first == "--models_dir") {
            models_dir = opt.second;
        } else if (opt.first == "--det_path") {
            det_path = opt.second;
        } else if (opt.first == "--cls_path") {
            cls_path = opt.second;
        } else if (opt.first == "--rec_path") {
            rec_path = opt.second;
        } else if (opt.first == "--keys_path") {
            keys_path = opt.second;
        } else if (opt.first == "--image_path") {
            image_path = opt.second;
        } else if (opt.first == "--num_threads") {
            num_threads = std::stoi(opt.second);
        } else if (opt.first == "--padding") {
            padding = std::stoi(opt.second);
        } else if (opt.first == "--max_side_len") {
            max_side_len = std::stoi(opt.second);
        } else if (opt.first == "--box_score_threshold") {
            box_score_threshold = std::stof(opt.second);
        } else if (opt.first == "--box_threshold") {
            box_threshold = std::stof(opt.second);
        } else if (opt.first == "--unclip_ratio") {
            unclip_ratio = std::stof(opt.second);
        } else if (opt.first == "--cal_angle") {
            cal_angle = opt.second == "true";
        } else if (opt.first == "--cal_most_angle") {
            cal_most_angle = opt.second == "true";
        } else {
            std::cerr << "Unknown option: " << opt.first << std::endl;
            return -1;
        }
    }

    // 模型参数检查
    if (models_dir.empty()) {
        std::cerr << "models_dir is empty" << std::endl;
        return -1;
    }

    if (det_path.empty()) det_path = utils::FileUtils::JoinPath(models_dir, "det.onnx");
    if (cls_path.empty()) cls_path = utils::FileUtils::JoinPath(models_dir, "cls.onnx");
    if (rec_path.empty()) rec_path = utils::FileUtils::JoinPath(models_dir, "rec.onnx");
    if (keys_path.empty()) keys_path = utils::FileUtils::JoinPath(models_dir, "keys.txt");

    if (!utils::FileUtils::IsFileExist(det_path)) {
        std::cerr << "det_path not found: " << det_path << std::endl;
        return -1;
    }

    if (!utils::FileUtils::IsFileExist(cls_path)) {
        std::cerr << "cls_path not found: " << cls_path << std::endl;
        return -1;
    }

    if (!utils::FileUtils::IsFileExist(rec_path)) {
        std::cerr << "rec_path not found: " << rec_path << std::endl;
        return -1;
    }

    if (!utils::FileUtils::IsFileExist(keys_path)) {
        std::cerr << "keys_path not found: " << keys_path << std::endl;
        return -1;
    }

    // 图像参数检查
    if (image_path.empty()) {
        std::cerr << "image_path is empty" << std::endl;
        return -1;
    }
    if (!utils::FileUtils::IsFileExist(image_path)) {
        std::cerr << "image_path not found: " << image_path << std::endl;
        return -1;
    }

    // 初始化 OCR 模型
    model::OcrLite ocr_lite;
    ocr_lite.Init(det_path, cls_path, rec_path, keys_path);
    ocr_lite.SetNumThreads(num_threads);
    if (opt_map.count("--output_console")) {
        ocr_lite.SetOutputConsole(true);
    }
    if (opt_map.count("--output_part_image")) {
        ocr_lite.SetOutputPartImage(true);
    }
    if (opt_map.count("--output_result_text")) {
        ocr_lite.SetOutputResultText(true);
    }
    if (opt_map.count("--output_result_image")) {
        ocr_lite.SetOutputResultImage(true);
    }

    // LOG_INFO

    double sum_det_time = 0.0;
    double sum_full_time = 0.0;

    // 读取图像
    if (utils::FileUtils::IsDirectory(image_path)) {
        std::vector<std::string> files;
        utils::FileUtils::ListDir(image_path, files);
        for (auto &image_name : files) {
            base::OcrResult result = ocr_lite.Process(image_dir, image_name, padding, max_side_len, box_score_threshold, box_threshold, unclip_ratio, cal_angle, cal_most_angle);
            // LOG_INFO
            std::cout << "det_time: " << result.det_time << " full_time: " << result.full_time << std::endl;

            sum_det_time += result.det_time;
            sum_full_time += result.full_time;
        }
    } else {
        image_dir = utils::FileUtils::GetDirName(image_path);
        std::string image_name = utils::FileUtils::GetFileName(image_path);

        base::OcrResult result = ocr_lite.Process(image_dir, image_name, padding, max_side_len, box_score_threshold, box_threshold, unclip_ratio, cal_angle, cal_most_angle);
        // LOG_INFO
        std::cout << "det_time: " << result.det_time << " full_time: " << result.full_time << std::endl;

        sum_det_time += result.det_time;
        sum_full_time += result.full_time;
    }

    // LOG_INFO
    std::cout << "=====Result=====" << std::endl;
    std::cout << "sum_det_time: " << sum_det_time << " sum_full_time: " << sum_full_time << std::endl;
    return 0;
}