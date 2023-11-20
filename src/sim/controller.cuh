#pragma once

#include <vector>
#include "point.cuh"
#include "modifierI.cuh"
#include "math/floating.h"
#include "colliders/cuboid.cuh"

namespace waterSim::sim{
    class controller{
    public:
        controller(size_t pointCount, float radius, vec3 domainSize);
        ~controller();

        void addModifier(modifierI *m);
        void removeModifier(modifierI *m);

        void syncDeviceToHost();
        void syncHostToDevice();
    private:
        point *pointsHost = nullptr;
        vec3 *pointsPosHostActive;
        point *pointsDevice = nullptr;
        std::vector<modifierI*> modifiersHost;
        modifierI** modifiersDevice = nullptr;
        size_t modifierArraySize = 0;
        size_t pointCount;

        colliders::cuboidHollow simulationDomain;

        [[noreturn]] void mainLoop();

        void step();
        void runModifiers();
        void runCollision();
        void updateGraphics();
    };
}