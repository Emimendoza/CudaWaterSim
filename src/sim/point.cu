#include "point.cuh"

namespace waterSim::sim {
    __host__ __device__ point::point(vec3 pos, vec3 vel, vec3 acc, float radius) {
        this->pos = pos;
        this->vel = vel;
        this->acc = acc;
        this->radius = radius;
        this->active = false;
    }

    __host__ __device__ point::point() {
        this->pos = vec3(0, 0, 0);
        this->vel = vec3(0, 0, 0);
        this->acc = vec3(0, 0, 0);
        this->radius = 0;
        this->active = false;
    }

    __host__ __device__ point::point(float radius) {
        this->pos = vec3(0, 0, 0);
        this->vel = vec3(0, 0, 0);
        this->acc = vec3(0, 0, 0);
        this->radius = radius;
        this->active = false;
    }
}