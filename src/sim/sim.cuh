#pragma once
#include <cuda_runtime.h>
#include "point.cuh"
#include "modifier.cuh"
#include "floating.h"


namespace waterSim::sim{
    constexpr size_t BLOCK_SIZE = 256;
    __global__ void modifyPoints(point *points, modifier **modifiers, size_t modifierCount, size_t pointCount);
    __global__ void velocityVerlet(point *points, size_t pointCount, FLOAT dt);
}