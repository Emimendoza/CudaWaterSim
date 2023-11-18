#include "sim.cuh"
#include "floating.h"

namespace waterSim::sim{
    __global__ void modifyPoints(point *points, modifier **modifiers, size_t modifierCount, size_t pointCount){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= pointCount) return;
        for (size_t j = 0; j < modifierCount; j++){
            modifiers[j]->modify(points[i]);
        }
    }

    __global__ void velocityVerlet(point *points, size_t pointCount, FLOAT dt){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= pointCount or !points[i].active) return;

    }
}
