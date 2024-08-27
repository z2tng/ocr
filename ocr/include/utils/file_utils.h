#pragma once

#include <string>
#include <vector>
#include <sys/stat.h>

namespace utils {

class FileUtils {
public:
    // 连接目录和文件名
    static std::string JoinPath(const std::string &dir, const std::string &file);
    
    static void ListDir(const std::string &path, std::vector<std::string> &files);

    static inline bool IsFileExist(const std::string &file) {
        struct stat buffer;
        return stat(file.c_str(), &buffer) == 0;
    }

    static inline bool IsDirectory(const std::string &path) {
        struct stat buffer;
        if (stat(path.c_str(), &buffer) == 0) {
            return S_ISDIR(buffer.st_mode);
        }
        return false;
    }

    static inline std::string GetDirName(const std::string &path) {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return path;
        }
        return path.substr(0, pos);
    }

    static inline std::string GetFileName(const std::string &path) {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return path;
        }
        return path.substr(pos + 1);
    }
};
    
} // namespace utils
