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
        __host__ __device__ vec3 nearestNonColliding(const point& p) const override{
            vec3 localPos = rot.rotate(p.pos - pos);
            return pos + rot.rotate(vec3(localPos.x, localPos.y, localPos.z).normalize() * (size / 2 + p.radius));
        }
        __device__ vec3 getRandomPoint(curandState& state) const override{
            // This should return a random point within the cuboid
            FLOAT x = UNIFORM_RANDOM(&state) * size.x - size.x / 2;
            FLOAT y = UNIFORM_RANDOM(&state) * size.y - size.y / 2;
            FLOAT z = UNIFORM_RANDOM(&state) * size.z - size.z / 2;
            return pos + rot.rotate(vec3(x, y, z));
        }
        __host__ __device__ vec3 tangent(const vec3& p) const override{
            vec3 localPos = rot.rotate(p - pos);
            vec3 absLocalPos = localPos.abs();

            // Calculate the normal vector of the nearest face
            vec3 normal;
            if (absLocalPos.x > absLocalPos.y && absLocalPos.x > absLocalPos.z) {
                normal = vec3(1, 0, 0);
            } else if (absLocalPos.y > absLocalPos.z) {
                normal = vec3(0, 1, 0);
            } else {
                normal = vec3(0, 0, 1);
            }

            // Project p onto the plane of the nearest face
            vec3 projectedP = p - normal * p.dot(normal);

            // Calculate a vector that is orthogonal to the normal vector and the projected p
            vec3 tangent = normal.cross(projectedP);

            // Scale the tangent vector to have the same length as p
            tangent = tangent.normalize() * p.length();

            // Rotate the tangent vector back to the global coordinate system
            return rot.rotate(tangent);
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

        __host__ __device__ vec3 nearestNonColliding(const point& p) const override{
            vec3 localPos = rot.rotate(p.pos - pos);
            // This point could be INSIDE the cuboid
            vec3 direction = vec3(localPos.x, localPos.y, localPos.z).normalize();
            return pos + rot.rotate(direction.hadamard(size / 2) + direction * p.radius);
        }
        __device__ vec3 getRandomPoint(curandState& state) const override{
            // First we get a random point within the cuboid
            FLOAT x = UNIFORM_RANDOM(&state) * size.x - size.x / 2;
            FLOAT y = UNIFORM_RANDOM(&state) * size.y - size.y / 2;
            FLOAT z = UNIFORM_RANDOM(&state) * size.z - size.z / 2;
            // We then tangent it onto the surface of the cuboid
            vec3 localPos = rot.rotate(vec3(x, y, z));
            vec3 direction = vec3(localPos.x, localPos.y, localPos.z).normalize();
            return pos + rot.rotate(direction.hadamard(size / 2));
        }
        __host__ __device__ vec3 tangent(const vec3& p) const override{
            vec3 localPos = rot.rotate(p - pos);
            vec3 absLocalPos = localPos.abs();

            // Calculate the normal vector of the nearest face
            vec3 normal;
            if (absLocalPos.x > absLocalPos.y && absLocalPos.x > absLocalPos.z) {
                normal = vec3(1, 0, 0);
            } else if (absLocalPos.y > absLocalPos.z) {
                normal = vec3(0, 1, 0);
            } else {
                normal = vec3(0, 0, 1);
            }

            // Project p onto the plane of the nearest face
            vec3 projectedP = p - normal * p.dot(normal);

            // Calculate a vector that is orthogonal to the normal vector and the projected p
            vec3 tangent = normal.cross(projectedP);

            // Scale the tangent vector to have the same length as p
            tangent = tangent.normalize() * p.length();

            // Rotate the tangent vector back to the global coordinate system
            return rot.rotate(tangent);
        }
    };
}