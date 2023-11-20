#pragma once

#include "../collisionI.cuh"
#include "../point.cuh"
#include "../math/floating.h"

namespace waterSim::sim::colliders{
    class sphericalSolid : public collisionI{
    public:
        FLOAT radius;
        explicit sphericalSolid(vec3 pos, FLOAT radius) : collisionI(pos), radius(radius) {}
        __host__ __device__ bool isColliding(const point& p) const override{
            return (p.pos - pos).length() < radius + p.radius;
        }
    };

    class sphericalHollow : public collisionI{
    public:
        FLOAT radius;
        explicit sphericalHollow(vec3 pos, FLOAT radius) : collisionI(pos), radius(radius) {}
        __host__ __device__ bool isColliding(const point& p) const override{
            FLOAT distance = (p.pos - pos).length();
            return distance > radius - p.radius && distance < radius + p.radius;
        }
    };
}