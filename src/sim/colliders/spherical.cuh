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
        __host__ __device__ vec3 nearestNonColliding(const point& p) const override{
            return pos + (p.pos - pos).normalize() * (radius + p.radius);
        }
        __device__ vec3 getRandomPoint(curandState& state) const override{
            // This should return a random point within the sphere
            FLOAT theta = UNIFORM_RANDOM(&state) * 2 * M_PI;
            FLOAT phi = UNIFORM_RANDOM(&state) * M_PI;
            FLOAT r = UNIFORM_RANDOM(&state) * radius;
            return pos + vec3(SIN(phi) * COS(theta), SIN(phi) * SIN(theta), COS(phi)) * r;
        }
        __host__ __device__ vec3 tangent(const vec3& p) const override{
            // Calculate the radius vector
            vec3 radiusVector = p - pos;

            // Calculate the normal vector of the sphere at the point of contact
            vec3 normal = radiusVector.normalize();

            // Project p onto the tangent plane
            vec3 projectedP = p - normal * p.dot(normal);

            // Calculate the tangent vector
            vec3 tangent = radiusVector.cross(projectedP);

            // Scale the tangent vector to have the same length as p
            tangent = tangent.normalize() * p.length();

            return tangent;
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
        __host__ __device__ vec3 nearestNonColliding(const point& p) const override{
            // This point could be INSIDE the sphere
            vec3 direction = (p.pos - pos).normalize();
            return pos + direction * radius + direction * p.radius;
        }
        __device__ vec3 getRandomPoint(curandState& state) const override{
            // This should return a random point touching the surface of the sphere
            FLOAT theta = UNIFORM_RANDOM(&state) * 2 * M_PI;
            FLOAT phi = UNIFORM_RANDOM(&state) * M_PI;
            return pos + vec3(SIN(phi) * COS(theta), SIN(phi) * SIN(theta), COS(phi)) * radius;
        }
        __host__ __device__ vec3 tangent(const vec3& p) const override{
            // Calculate the radius vector
            vec3 radiusVector = p - pos;

            // Calculate the normal vector of the sphere at the point of contact
            vec3 normal = radiusVector.normalize();

            // Project p onto the tangent plane
            vec3 projectedP = p - normal * p.dot(normal);

            // Calculate the tangent vector
            vec3 tangent = radiusVector.cross(projectedP);

            // Scale the tangent vector to have the same length as p
            tangent = tangent.normalize() * p.length();

            return tangent;
        }
    };
}