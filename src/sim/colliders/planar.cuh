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

        __host__ __device__ vec3 nearestNonColliding(const point& p) const override{
            vec3 localPos = rot.rotate(p.pos - pos);
            return pos + rot.rotate(vec3(localPos.x, localPos.y, 0).normalize() * (width / 2 + p.radius) + vec3(0, 0, localPos.z));
        }

        __device__ vec3 getRandomPoint(curandState& state) const override{
            // This should return a random point within the plane
            FLOAT x = UNIFORM_RANDOM(&state) * width - width / 2;
            FLOAT y = UNIFORM_RANDOM(&state) * height - height / 2;
            return pos + rot.rotate(vec3(x, y, 0));
        }
    };
}