#pragma once
#include "vec3.cuh"
#include "floating.h"

namespace waterSim::sim {
    class point {
    public:
        vec3 pos;
        vec3 vel;
        vec3 prevVel;
        vec3 prevAcc;
        FLOAT radius;
        bool active;
        __host__ __device__ point(vec3 pos, vec3 vel, FLOAT radius);
        __host__ __device__ explicit point(FLOAT radius);
        __host__ __device__ point();
    };
}
