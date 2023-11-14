#pragma once

namespace waterSim{
    class [[maybe_unused]] vec3{
        public:
            float x, y, z;
            __host__ __device__ vec3(float x, float y, float z);
            __host__ __device__ vec3();
            __host__ __device__ vec3 operator+(const vec3 &other) const;
            __host__ __device__ vec3 operator-(const vec3 &other) const;
            __host__ __device__ vec3 operator*(const float &other) const;
            __host__ __device__ vec3 operator/(const float &other) const;
            __host__ __device__ vec3 operator+=(const vec3 &other);
            __host__ __device__ vec3 operator-=(const vec3 &other);
            __host__ __device__ vec3 operator*=(const float &other);
            __host__ __device__ vec3 operator/=(const float &other);
            [[maybe_unused]] __host__ __device__ vec3 hadamard(const vec3 &other) const;
            [[maybe_unused]] __host__ __device__ float dot(const vec3 &other) const;
            [[maybe_unused]] __host__ __device__ vec3 cross(const vec3 &other) const;
            __host__ __device__ float length() const;
            [[maybe_unused]] __host__ __device__ vec3 normalize() const;
            __host__ __device__ vec3 operator-() const;
    };
}