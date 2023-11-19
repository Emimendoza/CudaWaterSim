#pragma once

#include <vector>
#include "point.cuh"
#include "modifierI.cuh"
#include "floating.h"

namespace waterSim::sim{
    class controller{
    public:
        controller(size_t pointCount, float radius);
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

        [[noreturn]] void mainLoop();

        void step();
        void runModifiers();
        void runGravity();
        void runCollision();
        void updateGraphics();
    };
}