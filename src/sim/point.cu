#include "point.cuh"

namespace waterSim::sim {
    __host__ __device__ point::point(vec3 pos, vec3 vel, FLOAT radius) {
        this->pos = pos;
        this->vel = vel;
        this->prevVel = vel;
        this->prevAcc = vec3(0, 0, 0);
        this->radius = radius;
        this->active = false;
    }

    __host__ __device__ point::point() {
        this->pos = vec3(0, 0, 0);
        this->vel = vec3(0, 0, 0);
        this->prevVel = vec3(0, 0, 0);
        this->prevAcc = vec3(0, 0, 0);
        this->radius = 0;
        this->active = false;
    }

    __host__ __device__ point::point(FLOAT radius) {
        this->pos = vec3(0, 0, 0);
        this->vel = vec3(0, 0, 0);
        this->prevVel = vec3(0, 0, 0);
        this->prevAcc = vec3(0, 0, 0);
        this->radius = radius;
        this->active = false;
    }

    __host__ vec3Primitive point::getPositionPrimitive() const {
        return this->pos.getPrimitive();
    }
}