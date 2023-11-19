#pragma once
#include "collisionI.cuh"

namespace waterSim::sim{
    class modifierI {
    public:
        collisionI *collider{};
        virtual __host__ __device__ void modify(point& p) = 0;
    };
}

