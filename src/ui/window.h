#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <atomic>
#include <thread>

namespace waterSim::ui {
    class window {
    public:
        window();
        window(int h, int w);
        ~window();
        void run();
        [[nodiscard]] bool isRunning() const { return running; }
        [[nodiscard]] std::atomic<bool>& getRunningObj() {return running;}
    private:
        std::atomic<bool> running{};
        int width, height;
        SDL_Window* windowPtr;

        void handleEvent(const SDL_Event& event);
        void mainLoop();
    };
}