#pragma once

#include "../collisionI.cuh"
#include "../point.cuh"

namespace waterSim::sim::colliders{
    class always : public collisionI{
    public:
        explicit always(vec3 pos) : collisionI(pos){}
        __host__ __device__ bool isColliding(const point& p) const override{
            return true;
        }
        __host__ __device__ vec3 nearestNonColliding(const point& p) const override{
            return p.pos;
        }
         __device__ vec3 getRandomPoint(curandStateXORWOW& state) const override{
            return pos;
        }
    };
}