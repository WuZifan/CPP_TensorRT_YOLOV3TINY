SET(common_dir /usr/src/tensorrt/samples/common)
INCLUDE_DIRECTORIES(${common_dir})

SET(cuda_dir /usr/local/cuda-10.0/targets/aarch64-linux/include)
INCLUDE_DIRECTORIES(${cuda_dir})

set(ONNX_PARSE /home/webank/tqiu/env/onnx-tensorrt-release-6.0/)
INCLUDE_DIRECTORIES(${ONNX_PARSE})

SET(LOG_CPP /usr/src/tensorrt/samples/common/logger.cpp)

find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )

SET(LOCAL_UTILS /home/webank/07_clion/TensorRT/utils)
INCLUDE_DIRECTORIES(${LOCAL_UTILS})
AUX_SOURCE_DIRECTORY(/home/webank/07_clion/TensorRT/utils UTILS_SRC)
MESSAGE( ${UTILS_SRC})

ADD_EXECUTABLE(trt_test main.cpp ${LOG_CPP} ${UTILS_SRC})

find_library(LIBONNX_PATH nvonnxparser /home/webank/tqiu/env/onnx-tensorrt-release-6.0/build)
TARGET_LINK_LIBRARIES(trt_test ${LIBONNX_PATH})

find_library(LIBNVINFER nvinfer /usr/lib/aarch64-linux-gnu/)
TARGET_LINK_LIBRARIES(trt_test ${LIBNVINFER})

find_library(LIBCUDART cudart /usr/local/cuda-10.0/lib64/)
TARGET_LINK_LIBRARIES(trt_test ${LIBCUDART})

target_link_libraries(trt_test ${OpenCV_LIBS})
