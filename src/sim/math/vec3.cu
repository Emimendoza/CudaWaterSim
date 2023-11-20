#include "vec3.cuh"
#include <cmath>


namespace waterSim::sim{
    __host__ __device__ vec3::vec3(FLOAT x, FLOAT y, FLOAT z){
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
    template<typename T>
    __host__ __device__ vec3 vec3::operator*(const T &other) const{
        return {this->x * other, this->y * other, this->z * other};
    }
    template<typename T>
    __host__ __device__ vec3 vec3::operator/(const T &other) const{
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
    template<typename T>
    __host__ __device__ vec3 vec3::operator*=(const T &other){
        this->x *= other;
        this->y *= other;
        this->z *= other;
        return *this;
    }
    template<typename T>
    __host__ __device__ vec3 vec3::operator/=(const T &other){
        this->x /= other;
        this->y /= other;
        this->z /= other;
        return *this;
    }

    [[maybe_unused]] __host__ __device__ vec3 vec3::hadamard(const vec3 &other) const{
        return {this->x * other.x, this->y * other.y, this->z * other.z};
    }

    [[maybe_unused]] __host__ __device__ FLOAT vec3::dot(const vec3 &other) const{
        return this->x * other.x + this->y * other.y + this->z * other.z;
    }

    [[maybe_unused]] __host__ __device__ vec3 vec3::cross(const vec3 &other) const{
        return {this->y * other.z - this->z * other.y, this->z * other.x - this->x * other.z, this->x * other.y - this->y * other.x};
    }
    __host__ __device__ FLOAT vec3::length() const{
        return SQRT(this->x * this->x + this->y * this->y + this->z * this->z);
    }

    [[maybe_unused]] __host__ __device__ vec3 vec3::normalize() const{
        return *this / this->length();
    }
    __host__ __device__ vec3 vec3::operator-() const{
        return {-this->x, -this->y, -this->z};
    }

    __host__ __device__ FLOAT vec3::sum() const {
        return this->x + this->y + this->z;
    }

    __host__ __device__ vec3 vec3::abs() const {
        return {ABS(this->x), ABS(this->y), ABS(this->z)};
    }

    template<typename T>
    __host__ __device__ vec3 vec3::operator+(const T &other) const {
        return {this->x + other, this->y + other, this->z + other};
    }
    template<typename T>
    __host__ __device__ vec3 vec3::operator-(const T &other) const {
        return {this->x - other, this->y - other, this->z - other};
    }
    template<typename T>
    __host__ __device__ vec3 vec3::operator+=(const T &other) const {
        this->x += other;
        this->y += other;
        this->z += other;
        return *this;
    }
    template<typename T>
    __host__ __device__ vec3 vec3::operator-=(const T &other) const {
        this->x -= other;
        this->y -= other;
        this->z -= other;
        return *this;
    }
}