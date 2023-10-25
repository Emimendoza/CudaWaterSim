cmake_minimum_required(VERSION 3.26)
project(CudaWaterSim CUDA CXX C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

add_executable(CudaWaterSim main.cu)

set_target_properties(CudaWaterSim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
