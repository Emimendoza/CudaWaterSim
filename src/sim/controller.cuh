#pragma once

#include <vector>
#include "point.cuh"
#include "modifierI.cuh"
#include "math/floating.h"
#include "colliders/cuboid.cuh"
#include <atomic>
#include <thread>
#include <string>
#include <mutex>
#include <queue>
#include "constants.h"

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
        struct bakedPoint{
            explicit bakedPoint(const point& p) : pos(p.pos), active(p.active){}
            vec3 pos;
            bool active;
        };
        std::string bakePath;
        size_t bakedFrameCount = 0;
        size_t bakedFrameSkip = 0;
        std::atomic<bool> baking = false;

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

    template<typename T>
    class threadSafeQueue{
    private:
        std::queue<T> queue;
        std::mutex mutex;
    public:
        void push(T& t){
            mutex.lock();
            queue.push(t);
            mutex.unlock();
        }
        void push(T&& t){
            mutex.lock();
            queue.push(std::move(t));
            mutex.unlock();
        }
        T pop(){
            while(true){
                mutex.lock();
                if(queue.empty()){
                    mutex.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                } else {
                    break;
                }
            }
            T front = queue.front();
            queue.pop();
            mutex.unlock();
            return front;
        }
    };
}