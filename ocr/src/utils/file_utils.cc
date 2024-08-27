#include "utils/file_utils.h"

#include <iostream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace utils {

std::string FileUtils::JoinPath(const std::string &dir, const std::string &file) {
    if (dir.empty()) {
        return file;
    }
    if (file.empty()) {
        return dir;
    }
    if (dir.back() == '/' || dir.back() == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

void FileUtils::ListDir(const std::string &path, std::vector<std::string> &files) {
    try {
        for (const auto &entry : fs::directory_iterator(path)) {
            files.emplace_back(entry.path().string());
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "ListDir error: " << e.what() << std::endl;
    }
}

} // namespace utils
