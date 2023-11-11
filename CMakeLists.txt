cmake_minimum_required(VERSION 3.26)
project(CudaWaterSim CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

include(CheckIncludeFileCXX)
check_include_file_cxx(format HAS_CXX_FORMAT)

if (NOT HAS_CXX_FORMAT)
    message(STATUS "The <format> header is not available, cloning fmt from GitHub...")

    # Set the directory where you want to clone fmt
    set(FMT_INSTALL_DIR "${CMAKE_BINARY_DIR}/fmt")

    # Clone fmt from GitHub
    if (NOT EXISTS "${FMT_INSTALL_DIR}/CMakeLists.txt")
        message(STATUS "Cloning fmt from GitHub...")

        # Remove the fmt directory if it exists
        if (EXISTS "${FMT_INSTALL_DIR}")
            file(REMOVE_RECURSE "${FMT_INSTALL_DIR}")
        endif ()

        # Clone fmt from GitHub
        execute_process(
                COMMAND git clone https://github.com/fmtlib/fmt.git ${FMT_INSTALL_DIR}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                RESULT_VARIABLE GIT_RESULT
        )

        if (NOT GIT_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to clone fmt from GitHub")
        endif ()
    else ()
        message(STATUS "Using existing fmt repository.")
    endif ()
    # Add fmt as a subdirectory and build it
    add_subdirectory(${FMT_INSTALL_DIR} fmt_build)

else ()
    message(STATUS "Found <format> header. No need to clone fmt.")
endif ()

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)

add_subdirectory(bgfx.cmake)

add_executable(CudaWaterSim src/utils/utils.h main.cu
        src/utils/utils.cu
        src/utils/utils.cpp)

set_target_properties(CudaWaterSim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
if(NOT HAS_CXX_FORMAT)
    target_link_libraries(CudaWaterSim fmt)
endif()
