#pragma once

#include <string>

namespace utils {

class FileUtils {
public:
    // 连接目录和文件名
    static std::string JoinPath(const std::string &dir, const std::string &file);
};
    
} // namespace utils
