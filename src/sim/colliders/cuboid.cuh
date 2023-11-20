#pragma once

#include "../collisionI.cuh"
#include "../math/quaternion.cuh"

namespace waterSim::sim::colliders{
    class cuboidSolid : public collisionI{
    public:
        vec3 size;
        quaternion rot;
        explicit cuboidSolid(vec3 pos, vec3 size, quaternion rot) : collisionI(pos), size(size), rot(rot) {}
        __host__ __device__ bool isColliding(const point& p) const override{
            vec3 localPos = rot.rotate(p.pos - pos);
            return localPos.x > -size.x / 2 - p.radius && localPos.x < size.x / 2 + p.radius &&
                   localPos.y > -size.y / 2 - p.radius && localPos.y < size.y / 2 + p.radius &&
                   localPos.z > -size.z / 2 - p.radius && localPos.z < size.z / 2 + p.radius;
        }
    };

    class cuboidHollow : public collisionI{
    public:
        vec3 size;
        quaternion rot;
        explicit cuboidHollow(vec3 pos, vec3 size, quaternion rot) : collisionI(pos), size(size), rot(rot) {}
        __host__ __device__ bool isColliding(const point& p) const override{
            vec3 localPos = rot.rotate(p.pos - pos);
            // Only collide if point is touching the surface of the cuboid. The point can exist inside the cuboid.
            return (localPos.x > -size.x / 2 - p.radius && localPos.x < size.x / 2 + p.radius &&
                    localPos.y > -size.y / 2 - p.radius && localPos.y < size.y / 2 + p.radius &&
                    localPos.z > -size.z / 2 - p.radius && localPos.z < size.z / 2 + p.radius) &&
                   (localPos.x < -size.x / 2 + p.radius || localPos.x > size.x / 2 - p.radius ||
                    localPos.y < -size.y / 2 + p.radius || localPos.y > size.y / 2 - p.radius ||
                    localPos.z < -size.z / 2 + p.radius || localPos.z > size.z / 2 - p.radius);
        }
    };
}