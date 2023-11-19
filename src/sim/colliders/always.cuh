#pragma once

#include "../collisionI.cuh"
#include "../point.cuh"

namespace waterSim::sim::colliders{
    class always : public collisionI{
    public:
        __host__ __device__ bool isColliding(const point& p) const override{
            return true;
        }
    };
}