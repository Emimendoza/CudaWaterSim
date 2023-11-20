#pragma once

#include <curand_kernel.h>
#include "math/vec3.cuh"
#include "point.cuh"

namespace waterSim::sim{
    class collisionI{
    public:
        vec3 pos;
        __host__ __device__ explicit collisionI(vec3 pos) : pos(pos){}
        [[maybe_unused]] [[nodiscard]] virtual __host__ __device__ bool isColliding(const point& p) const = 0;
        [[maybe_unused]] [[nodiscard]] virtual __host__ __device__ vec3 nearestNonColliding(const point& p) const = 0;
        [[maybe_unused]] [[nodiscard]] virtual __device__ vec3 getRandomPoint(curandState &state) const = 0;
    };
}