cmake_minimum_required(VERSION 3.3)
project(SpMM VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_compile_options(-w -O3 --use_fast_math)
set(CMAKE_CUDA_ARCHITECTURES "80")

find_package(CUDAToolkit)

add_executable(test_spmm tests/test_spmm.cu)

target_include_directories(test_spmm PUBLIC include)

target_link_libraries(test_spmm PUBLIC CUDA::cusparse CUDA::cudart CUDA::cublas )

