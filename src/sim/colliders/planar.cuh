#pragma once

#include "../math/floating.h"
#include "../point.cuh"
#include "../collisionI.cuh"
#include "../math/quaternion.cuh"
#include "../math/vec2.cuh"

namespace waterSim::sim::colliders{
    class planar : public collisionI{
    public:
        vec2 size;
        quaternion rot;
        explicit planar(vec3 pos, vec2 size, quaternion rot) : collisionI(pos), size(size), rot(rot) {}
        __host__ __device__ bool isColliding(const point& p) const override{
            vec3 localPos = rot.rotate(p.pos - pos);
            return localPos.x > -size.x / 2 - p.radius && localPos.x < size.x / 2 + p.radius &&
                   localPos.y > -size.y / 2 - p.radius && localPos.y < size.y / 2 + p.radius;
        }

        __host__ __device__ vec3 nearestNonColliding(const point& p) const override{
            vec3 localPos = rot.rotate(p.pos - pos);
            return pos + rot.rotate(vec3(localPos.x, localPos.y, 0).normalize() * (size.x / 2 + p.radius) + vec3(0, 0, localPos.z));
        }

        __device__ vec3 getRandomPoint(curandState& state) const override{
            // This should return a random point within the plane
            FLOAT x = UNIFORM_RANDOM(&state) * size.x - size.x / 2;
            FLOAT y = UNIFORM_RANDOM(&state) * size.y - size.y / 2;
            return pos + rot.rotate(vec3(x, y, 0));
        }

        __host__ __device__ vec3 tangent(const vec3& p) const override{
            vec3 normal = vec3(0, 0, 1);

            vec3 projectedP = p - normal * p.dot(normal);
            vec3 tangent = normal.cross(projectedP);
            tangent = tangent.normalize() * p.length();
            return rot.rotate(tangent);
        }
    };
}