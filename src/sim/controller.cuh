#pragma once

#include "point.cuh"
#include "modifier.cuh"

namespace waterSim::sim{
    class controller{
    public:
        controller(size_t pointCount, float radius);
        ~controller();

        void addModifier(modifier *m);
        void removeModifier(modifier *m);

        void syncDeviceToHost();
        void syncHostToDevice();
    private:
        point *pointsHost;
        vec3 *pointsPosHostActive;
        point *pointsDevice;
        modifier **modifiersHost;
        modifier **modifiersDevice;
        size_t modifierCount;
        size_t pointCount;

        void step();
    };
}