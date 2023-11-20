#include "vec2.cuh"

namespace waterSim::sim{
    __host__ __device__ vec2::vec2(FLOAT x, FLOAT y) {
        this->x = x;
        this->y = y;
    }

    __host__ __device__ vec2::vec2() {
        this->x = 0;
        this->y = 0;
    }
    
    __host__ __device__ vec2 vec2::operator+(const vec2 &other) const {
        return {this->x + other.x, this->y + other.y};
    }
    
    __host__ __device__ vec2 vec2::operator-(const vec2 &other) const {
        return {this->x - other.x, this->y - other.y};
    }
    
    __host__ __device__ vec2 vec2::operator-() const {
        return {-this->x, -this->y};
    }
    
    
    __host__ __device__ vec2 vec2::operator+(const FLOAT &other) const {
        return {this->x + other, this->y + other};
    }
    
    
    __host__ __device__ vec2 vec2::operator-(const FLOAT &other) const {
        return {this->x - other, this->y - other};
    }
    
    
    __host__ __device__ vec2 vec2::operator*(const FLOAT &other) const {
        return {this->x * other, this->y * other};
    }
    
    
    __host__ __device__ vec2 vec2::operator/(const FLOAT &other) const {
        return {this->x / other, this->y / other};
    }
    
    
    __host__ __device__ vec2 vec2::operator+=(const FLOAT &other) const {
        return {this->x + other, this->y + other};
    }
    
    
    __host__ __device__ vec2 vec2::operator-=(const FLOAT &other) const {
        return {this->x - other, this->y - other};
    }
    
    __host__ __device__ vec2 vec2::operator+=(const vec2 &other) {
        this->x += other.x;
        this->y += other.y;
        return *this;
    }
    
    __host__ __device__ vec2 vec2::operator-=(const vec2 &other) {
        this->x -= other.x;
        this->y -= other.y;
        return *this;
    }
    
    
    __host__ __device__ vec2 vec2::operator*=(const FLOAT &other) {
        this->x *= other;
        this->y *= other;
        return *this;
    }
    
    
    __host__ __device__ vec2 vec2::operator/=(const FLOAT &other) {
        this->x /= other;
        this->y /= other;
        return *this;
    }
    
    [[maybe_unused]] __host__ __device__ FLOAT vec2::sum() const {
        return this->x + this->y;
    }
    
    [[maybe_unused]] __host__ __device__ vec2 vec2::abs() const {
        return {fabsf(this->x), fabsf(this->y)};
    }
    
    [[maybe_unused]] __host__ __device__ vec2 vec2::hadamard(const vec2 &other) const {
        return {this->x * other.x, this->y * other.y};
    }
    
    [[maybe_unused]] __host__ __device__ FLOAT vec2::dot(const vec2 &other) const {
        return this->x * other.x + this->y * other.y;
    }
    
    [[maybe_unused]] __host__ __device__ FLOAT vec2::cross(const vec2 &other) const {
        return this->x * other.y - this->y * other.x;
    }
    
    [[maybe_unused]] __host__ __device__ FLOAT vec2::length() const {
        return sqrtf(this->x * this->x + this->y * this->y);
    }
    
    [[maybe_unused]] __host__ __device__ vec2 vec2::normalize() const {
        return *this / this->length();
    }

    __host__ __device__ vec3 vec2::cross(const vec3 &other) const {
        return {0, 0, this->x * other.y - this->y * other.x};
    }
}