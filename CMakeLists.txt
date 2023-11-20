cmake_minimum_required(VERSION 3.26)
project(CudaWaterSim CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Define if you want to use double precision
# add_compile_definitions(DOUBLE_PRECISION)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in release mode")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -O3 -Xcompiler -O3,-s,-flto")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -s -flto")
else()
    message(STATUS "Building in debug mode")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

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

add_executable(CudaWaterSim src/utils.h
        src/sim/sim.cu
        src/sim/sim.cuh
        src/sim/math/vec3.cu
        src/sim/math/vec3.cuh
        main.cu
        src/sim/point.cu
        src/sim/point.cuh
        src/sim/modifierI.cuh
        src/sim/collisionI.cuh
        src/sim/controller.cuh
        src/sim/controller.cu
        src/sim/math/floating.h
        src/sim/colliders/always.cuh
        src/sim/colliders/spherical.cuh
        src/sim/colliders/planar.cuh
        src/sim/math/quaternion.cuh
        src/sim/math/quaternion.cu
        src/sim/colliders/cuboid.cuh
        src/sim/math/vec2.cu
        src/sim/math/vec2.cuh)

set_target_properties(CudaWaterSim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
if(NOT HAS_CXX_FORMAT)
    target_link_libraries(CudaWaterSim fmt)
endif()
