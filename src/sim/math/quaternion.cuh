#pragma once

#include "floating.h"
#include "vec3.cuh"

namespace waterSim::sim{
    class quaternion{
    public:
        FLOAT w;
        vec3 v;
        __host__ __device__ quaternion(FLOAT w, vec3 v) : w(w), v(v){}
        __host__ __device__ quaternion() : w(1), v(vec3()){} // Identity quaternion
        __host__ __device__ quaternion(const vec3& axis, FLOAT angle);
        __host__ __device__ explicit quaternion(const vec3& euler);
        __host__ __device__ FLOAT toAxisAngle(vec3& axis) const;
        __host__ __device__ vec3 toEuler() const;
        __host__ __device__ quaternion conjugate() const;
        __host__ __device__ FLOAT length() const;
        __host__ __device__ quaternion normalize() const;
        __host__ __device__ quaternion multiply(const quaternion& other) const;
        __host__ __device__ vec3 rotate(const vec3& other) const;
        __host__ __device__ void rotateInPlace(vec3& other) const;

    };
}