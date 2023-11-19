#pragma once

#include "../floating.h"
#include "../point.cuh"
#include "../collisionI.cuh"
#include "../quaternion.cuh"

namespace waterSim::sim::colliders{
    class planar : public collisionI{
    public:
        FLOAT height, width;
        quaternion rot;
        explicit planar(FLOAT height, FLOAT width, quaternion rot) : height(height), width(width), rot(rot){}
        __host__ __device__ bool isColliding(const point& p) const override{
            return false; // TODO: Implement
        }
    };
}