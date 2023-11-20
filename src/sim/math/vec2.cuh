#pragma once
#include "floating.h"
#include "vec3.cuh"

namespace waterSim::sim{
    class vec2 {
    public:
        FLOAT x, y;
        __host__ __device__ vec2(FLOAT x, FLOAT y);
        __host__ __device__ vec2();
        __host__ __device__ vec2 operator+(const vec2 &other) const;
        __host__ __device__ vec2 operator-(const vec2 &other) const;
        __host__ __device__ vec2 operator-() const;
        __host__ __device__ vec2 operator+(const FLOAT &other) const;
        __host__ __device__ vec2 operator-(const FLOAT &other) const;
        __host__ __device__ vec2 operator*(const FLOAT &other) const;
        __host__ __device__ vec2 operator/(const FLOAT &other) const;
        __host__ __device__ vec2 operator+=(const FLOAT &other) const;
        __host__ __device__ vec2 operator-=(const FLOAT &other) const;
        __host__ __device__ vec2 operator+=(const vec2 &other);
        __host__ __device__ vec2 operator-=(const vec2 &other);
        __host__ __device__ vec2 operator*=(const FLOAT &other);
        __host__ __device__ vec2 operator/=(const FLOAT &other);
        /**
         * @return The sum of all components
         */
        [[maybe_unused]] __host__ __device__ FLOAT sum() const;
        [[maybe_unused]] __host__ __device__ vec2 abs() const;
        [[maybe_unused]] __host__ __device__ vec2 hadamard(const vec2 &other) const;
        [[maybe_unused]] __host__ __device__ FLOAT dot(const vec2 &other) const;
        [[maybe_unused]] __host__ __device__ FLOAT cross(const vec2 &other) const;
        [[maybe_unused]] __host__ __device__ vec3 cross(const vec3 &other) const;
        [[maybe_unused]] __host__ __device__ FLOAT length() const;
        [[maybe_unused]] __host__ __device__ vec2 normalize() const;
    };
}
