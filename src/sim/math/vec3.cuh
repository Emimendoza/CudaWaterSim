#pragma once

#include "floating.h"
#include "vec3Primitive.h"
namespace waterSim::sim{
    class vec3{
    private:
        vec3Primitive primitive;
    public:
        // make this.x and this.y and this.z actually be primitive.x, primitive.y, and primitive.z
        FLOAT &x = primitive.x, &y = primitive.y, &z = primitive.z;
        __host__ __device__ vec3(FLOAT x, FLOAT y, FLOAT z);
        __host__ __device__ vec3();
        __host__ __device__ vec3 operator+(const vec3 &other) const;
        __host__ __device__ vec3 operator-(const vec3 &other) const;
        __host__ __device__ vec3 operator-() const;
        __host__ __device__ vec3 operator+(const FLOAT &other) const;
        __host__ __device__ vec3 operator-(const FLOAT &other) const;
        __host__ __device__ vec3 operator*(const FLOAT &other) const;
        __host__ __device__ vec3 operator/(const FLOAT &other) const;
        __host__ __device__ vec3 operator+=(const FLOAT &other);
        __host__ __device__ vec3 operator-=(const FLOAT &other);
        __host__ __device__ vec3 operator+=(const vec3 &other);
        __host__ __device__ vec3 operator-=(const vec3 &other);
        __host__ __device__ vec3 operator*=(const FLOAT &other);
        __host__ __device__ vec3 operator/=(const FLOAT &other);
        __host__ __device__ vec3 operator=(const vec3& other);
        /**
         * @return The sum of all components
         */
        [[maybe_unused]] __host__ __device__ FLOAT sum() const;
        [[maybe_unused]] __host__ __device__ vec3 abs() const;
        [[maybe_unused]] __host__ __device__ vec3 hadamard(const vec3 &other) const;
        [[maybe_unused]] __host__ __device__ FLOAT dot(const vec3 &other) const;
        [[maybe_unused]] __host__ __device__ vec3 cross(const vec3 &other) const;
        [[maybe_unused]] __host__ __device__ FLOAT length() const;
        [[maybe_unused]] __host__ __device__ vec3 normalize() const;
        [[maybe_unused]] __host__ __device__ vec3Primitive getPrimitive() const;
    };
}