#pragma once

#include <vector>
#include "../point.cuh"
#include "../modifierI.cuh"
#include "../math/floating.h"
#include "../colliders/cuboid.cuh"
#include <atomic>
#include <thread>
#include <string>
#include <mutex>
#include <queue>
#include "../constants.h"
#include "threadSafeQueue.h"
#include "file.h"
#include "bakedPoint.h"

namespace waterSim::sim{
    class controller{
    public:
        controller(size_t pointCount, float radius, vec3 domainSize);
        ~controller();

        void addModifier(modifierI *m);
        void removeModifier(modifierI *m);

        void run(std::atomic<bool>& condition);
        void bake(size_t frameCount, size_t frameSkip = 0);

        void syncDeviceToHost();
        void syncHostToDevice();

        void pause();
        void resume();
        bool setBakePath(const std::string& path);
        [[nodiscard]] bool isPaused() const {return simPaused;}
    private:

        point* pointsBuffer[MAX_QUEUE_SIZE]{};
        threadSafeQueue<point*> pointersFree;
        threadSafeQueue<point*> pointersFilled;
        bakedPoint* bakedPointsBuffer;
        std::string bakePath;
        size_t bakedFrameCount = 0;
        size_t bakedFrameSkip = 0;
        std::atomic<bool> baking = false;

        void bakeFrameToFile();

        point *pointsHost = nullptr;
        vec3 *pointsPosHostActive;
        point *pointsDevice = nullptr;
        std::vector<modifierI*> modifiersHost;
        modifierI** modifiersDevice = nullptr;
        size_t modifierArraySize = 0;
        size_t pointCount;

        std::thread simThread{};
        std::atomic<bool> simRunning{};
        std::atomic<bool> simPaused{};
        bool simStarted = false;

        colliders::cuboidHollow simulationDomain;

        void mainLoop(std::atomic<bool>& condition);

        void step();
        void bakeFrame();
        void runModifiers();
        void runCollision();
        void updateGraphics();
    };


}