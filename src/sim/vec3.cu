#include "vec3.cuh"
#include <cmath>

namespace waterSim{
    __host__ __device__ vec3::vec3(float x, float y, float z){
        this->x = x;
        this->y = y;
        this->z = z;
    }
    __host__ __device__ vec3::vec3(){
        this->x = 0;
        this->y = 0;
        this->z = 0;
    }
    __host__ __device__ vec3 vec3::operator+(const vec3 &other) const{
        return {this->x + other.x, this->y + other.y, this->z + other.z};
    }
    __host__ __device__ vec3 vec3::operator-(const vec3 &other) const{
        return {this->x - other.x, this->y - other.y, this->z - other.z};
    }
    __host__ __device__ vec3 vec3::operator*(const float &other) const{
        return {this->x * other, this->y * other, this->z * other};
    }
    __host__ __device__ vec3 vec3::operator/(const float &other) const{
        return {this->x / other, this->y / other, this->z / other};
    }
    __host__ __device__ vec3 vec3::operator+=(const vec3 &other){
        this->x += other.x;
        this->y += other.y;
        this->z += other.z;
        return *this;
    }
    __host__ __device__ vec3 vec3::operator-=(const vec3 &other){
        this->x -= other.x;
        this->y -= other.y;
        this->z -= other.z;
        return *this;
    }
    __host__ __device__ vec3 vec3::operator*=(const float &other){
        this->x *= other;
        this->y *= other;
        this->z *= other;
        return *this;
    }
    __host__ __device__ vec3 vec3::operator/=(const float &other){
        this->x /= other;
        this->y /= other;
        this->z /= other;
        return *this;
    }

    [[maybe_unused]] __host__ __device__ vec3 vec3::hadamard(const vec3 &other) const{
        return {this->x * other.x, this->y * other.y, this->z * other.z};
    }

    [[maybe_unused]] __host__ __device__ float vec3::dot(const vec3 &other) const{
        return this->x * other.x + this->y * other.y + this->z * other.z;
    }

    [[maybe_unused]] __host__ __device__ vec3 vec3::cross(const vec3 &other) const{
        return {this->y * other.z - this->z * other.y, this->z * other.x - this->x * other.z, this->x * other.y - this->y * other.x};
    }
    __host__ __device__ float vec3::length() const{
        return sqrtf(this->x * this->x + this->y * this->y + this->z * this->z);
    }

    [[maybe_unused]] __host__ __device__ vec3 vec3::normalize() const{
        return *this / this->length();
    }
    __host__ __device__ vec3 vec3::operator-() const{
        return {-this->x, -this->y, -this->z};
    }
}