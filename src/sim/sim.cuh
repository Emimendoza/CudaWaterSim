#pragma once
#include <cuda_runtime.h>
#include "point.cuh"
#include "modifierI.cuh"
#include "math/floating.h"


namespace waterSim::sim{
    __global__ void modifyPoints(point *points, modifierI **modifiers, size_t modifierCount, size_t pointCount);
    __global__ void velocityVerlet(point *points, size_t pointCount, FLOAT dt);
}