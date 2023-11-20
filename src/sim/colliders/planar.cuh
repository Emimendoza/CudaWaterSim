#pragma once

#include "../math/floating.h"
#include "../point.cuh"
#include "../collisionI.cuh"
#include "../math/quaternion.cuh"

namespace waterSim::sim::colliders{
    class planar : public collisionI{
    public:
        FLOAT height, width;
        quaternion rot;
        explicit planar(vec3 pos, FLOAT height, FLOAT width, quaternion rot) : collisionI(pos), height(height),
                                                                               width(width), rot(rot) {}
        __host__ __device__ bool isColliding(const point& p) const override{
            vec3 localPos = rot.rotate(p.pos - pos);
            return localPos.x > -width / 2 - p.radius && localPos.x < width / 2 + p.radius &&
                   localPos.y > -height / 2 - p.radius && localPos.y < height / 2 + p.radius;
        }
    };
}