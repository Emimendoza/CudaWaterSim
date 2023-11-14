#pragma once
#include "vec3.cuh"

namespace waterSim::sim {
    class point {
    public:
        vec3 pos;
        vec3 vel;
        vec3 acc;
        float radius;
        bool active;
        __host__ __device__ point(vec3 pos, vec3 vel, vec3 acc, float radius);
        __host__ __device__ point(float radius);
        __host__ __device__ point();
    };

}
