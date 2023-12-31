#include "quaternion.cuh"
#include "floating.h"

namespace waterSim::sim{
    __host__ __device__ quaternion::quaternion(const vec3& axis, FLOAT angle) {
        FLOAT halfAngle = angle / 2;
        this->w = COS(halfAngle);
        FLOAT sinHalfAngle = SIN(halfAngle);
        this->v = axis * sinHalfAngle;
    }

    __host__ __device__ FLOAT quaternion::toAxisAngle(vec3 &axis) const {
        FLOAT angle = 2 * ACOS(this->w);
        FLOAT div = SQRT (1 - this->w * this->w);
        if (div == 0) {
            axis = vec3(1, 0, 0);
        } else {
            axis = this->v / div;
        }
        return angle;
    }

    __host__ __device__ quaternion::quaternion(const vec3 &euler) {
        // From https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        vec3 halfEuler = euler / 2;
        FLOAT cr = COS(halfEuler.x);
        FLOAT sr = SIN(halfEuler.x);
        FLOAT cp = COS(halfEuler.y);
        FLOAT sp = SIN(halfEuler.y);
        FLOAT cy = COS(halfEuler.z);
        FLOAT sy = SIN(halfEuler.z);
        this->w = cr * cp * cy + sr * sp * sy;
        this->v.x = sr * cp * cy - cr * sp * sy;
        this->v.y = cr * sp * cy + sr * cp * sy;
        this->v.z = cr * cp * sy - sr * sp * cy;
    }

    __host__ __device__ vec3 quaternion::toEuler() const {
        // From https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        vec3 euler;
        // Roll
        FLOAT sinrCosp = 2 * (this->w * this->v.x + this->v.y * this->v.z);
        FLOAT cosrCosp = 1 - 2 * (this->v.x * this->v.x + this->v.y * this->v.y);
        euler.x = ATAN2(sinrCosp, cosrCosp);

        // Pitch
        FLOAT sinp = 2 * (this->w * this->v.y + this->v.z * this->v.x);
        if (ABS(sinp) >= 1)
            euler.y = COPYSIGN(M_PI / 2, sinp); // use 90 degrees if out of range
        else
            euler.y = ASIN(sinp);

        // Yaw
        FLOAT sinyCosp = 2 * (this->w * this->v.z + this->v.x * this->v.y);
        FLOAT cosyCosp = 1 - 2 * (this->v.y * this->v.y + this->v.z * this->v.z);
        euler.z = ATAN2(sinyCosp, cosyCosp);

        return euler;
    }

    __host__ __device__ quaternion quaternion::conjugate() const {
        quaternion q = *this;
        q.v = -q.v;
        return q;
    }

    __host__ __device__ FLOAT quaternion::length() const {
        return SQRT(this->w * this->w + this->v.hadamard(this->v).sum());
    }

    __host__ __device__ quaternion quaternion::normalize() const {
        FLOAT len = this->length();
        return {this->w / len, this->v / len};
    }

    __host__ __device__ quaternion quaternion::multiply(const quaternion &other) const {
        quaternion q;
        q.w = this->w * other.w - this->v.dot(other.v);
        q.v =  other.v * this->w + this->v * other.w + this->v.cross(other.v);
        return q;
    }

    __host__ __device__ vec3 quaternion::rotate(const vec3 &other) const {
        quaternion q = *this;
        quaternion p(0, other);
        quaternion qInv = q.conjugate();
        quaternion rotated = q.multiply(p).multiply(qInv);
        return rotated.v;
    }

    __host__ __device__ void quaternion::rotateInPlace(vec3 &other) const {
        quaternion q = *this;
        quaternion p(0, other);
        quaternion qInv = q.conjugate();
        quaternion rotated = q.multiply(p).multiply(qInv);
        other = rotated.v;
    }
}