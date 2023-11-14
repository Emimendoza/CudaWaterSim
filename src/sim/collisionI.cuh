#pragma once
#include "vec3.cuh"
#include "point.cuh"

namespace waterSim::sim{
    class collisionI{
    public:
        vec3 pos;
        [[nodiscard]] virtual __host__ __device__ bool isColliding(const point& p) const = 0;
    };
}