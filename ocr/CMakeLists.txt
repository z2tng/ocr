# 项目定义
project(OcrLiteOnnx LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5.1)

# 项目目录设置
set(ROOT_DIR            ${CMAKE_CURRENT_LIST_DIR})
set(SRC_DIR             ${ROOT_DIR}/src)
set(INC_DIR             ${ROOT_DIR}/include)
set(LIB_DIR             /usr/local/lib)
set(EXEC_INSTALL_DIR    ${ROOT_DIR}/bin)
set(LIB_INSTALL_DIR     ${ROOT_DIR}/lib)

# 编译选项
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wall -Wno-unused-parameter -Wl,-rpath=${LIB_INSTALL_DIR}")

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

# 源文件收集
file (GLOB MODEL_SRC_FILE   ${SRC_DIR}/model/*.cc)
file (GLOB UTILS_SRC_FILE   ${SRC_DIR}/utils/*.cc)
set(MAIN_SRC_FILE           ${ROOT_DIR}/main.cc)

set(OCR_SRC ${MODEL_SRC_FILE}
            ${UTILS_SRC_FILE})

# 头文件路径
include_directories(${INC_DIR} ${OpenCV_INCLUDE_DIRS})
# 链接库路径
set(LINK_LIB onnxruntime)

# 生成静态库
add_library(ocr_static STATIC ${OCR_SRC})
set_target_properties(ocr_static PROPERTIES OUTPUT_NAME ocr)
target_link_libraries(ocr_static ${LINK_LIB} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# 生成动态库
add_library(ocr_shared SHARED ${OCR_SRC})
set_target_properties(ocr_shared PROPERTIES OUTPUT_NAME ocr)
target_link_libraries(ocr_shared ${LINK_LIB} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# 生成可执行文件
add_executable(OcrLiteOnnx ${MAIN_SRC_FILE})
target_link_libraries(OcrLiteOnnx ocr_static ${LINK_LIB} ${OpenCV_LIBS} OpenMP::OpenMP_CXX stdc++fs)

# 安装设置
install(TARGETS OcrLiteOnnx DESTINATION ${EXEC_INSTALL_DIR})
install(TARGETS ocr_static DESTINATION ${LIB_INSTALL_DIR})
install(TARGETS ocr_shared DESTINATION ${LIB_INSTALL_DIR})