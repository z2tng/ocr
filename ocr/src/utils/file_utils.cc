#include "utils/file_utils.h"

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
    
} // namespace utils
