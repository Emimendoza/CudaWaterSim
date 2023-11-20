#pragma once

#include "floating.h"
namespace waterSim{
    class [[maybe_unused]] vec3{
        public:
            FLOAT x, y, z;
            __host__ __device__ vec3(FLOAT x, FLOAT y, FLOAT z);
            __host__ __device__ vec3();
            __host__ __device__ vec3 operator+(const vec3 &other) const;
            __host__ __device__ vec3 operator-(const vec3 &other) const;
            template<typename T>
            __host__ __device__ vec3 operator+(const T &other) const;
            template<typename T>
            __host__ __device__ vec3 operator-(const T &other) const;
            template<typename T>
            __host__ __device__ vec3 operator*(const T &other) const;
            template<typename T>
            __host__ __device__ vec3 operator/(const T &other) const;
            template<typename T>
            __host__ __device__ vec3 operator+=(const T &other) const;
            template<typename T>
            __host__ __device__ vec3 operator-=(const T &other) const;
            __host__ __device__ vec3 operator+=(const vec3 &other);
            __host__ __device__ vec3 operator-=(const vec3 &other);
            template<typename T>
            __host__ __device__ vec3 operator*=(const T &other);
            template<typename T>
            __host__ __device__ vec3 operator/=(const T &other);
            /**
             * @return The sum of all components
             */
            [[maybe_unused]] __host__ __device__ FLOAT sum() const;
            [[maybe_unused]] __host__ __device__ vec3 hadamard(const vec3 &other) const;
            [[maybe_unused]] __host__ __device__ FLOAT dot(const vec3 &other) const;
            [[maybe_unused]] __host__ __device__ vec3 cross(const vec3 &other) const;
            __host__ __device__ FLOAT length() const;
            [[maybe_unused]] __host__ __device__ vec3 normalize() const;
            __host__ __device__ vec3 operator-() const;
    };
}