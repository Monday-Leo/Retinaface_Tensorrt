cmake_minimum_required(VERSION 2.6)

project(retinaface)

set(OpenCV_DIR "G:\\c++\\paddle_test\\opencv\\build")
set(TRT_DIR "G:\\c++\\TensorRT-8.2.1.8")

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})
cuda_add_library(decodeplugin STATIC  ${PROJECT_SOURCE_DIR}/decode.cu)
target_link_libraries(decodeplugin nvinfer cudart)

#change to your GPU own compute_XX
###########################################################################################
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75)
###########################################################################################

####
enable_language(CUDA)  
####
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_DIR}\\include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${TRT_DIR}\\lib)

add_executable(retina_r50 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/retina_r50.cpp)
target_link_libraries(retina_r50 nvinfer)
target_link_libraries(retina_r50 decodeplugin)
target_link_libraries(retina_r50 ${CUDA_LIBRARIES})   
target_link_libraries(retina_r50 Threads::Threads)       
target_link_libraries(retina_r50 ${OpenCV_LIBRARIES})

add_executable(retina_mnet ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/retina_mnet.cpp)
target_link_libraries(retina_mnet nvinfer)
target_link_libraries(retina_mnet decodeplugin)
target_link_libraries(retina_mnet ${CUDA_LIBRARIES})  
target_link_libraries(retina_mnet Threads::Threads)      
target_link_libraries(retina_mnet ${OpenCV_LIBRARIES})


