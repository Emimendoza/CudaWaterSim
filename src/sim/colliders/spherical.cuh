#pragma once

#include "../collisionI.cuh"
#include "../point.cuh"
#include "../floating.h"

namespace waterSim::sim::colliders{
    class spherical : public collisionI{
    public:
        FLOAT radius;
        explicit spherical(FLOAT radius) : radius(radius){}
        __host__ __device__ bool isColliding(const point& p) const override{
            return (p.pos - pos).length() < radius;
        }
    };
}