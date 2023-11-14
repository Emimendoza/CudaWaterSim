#include "sim.cuh"

namespace waterSim{
    __global__ void sum(const int *a, int *b, int n){
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid < n){
            b[tid] = a[tid] + b[tid];
        }
    }
}