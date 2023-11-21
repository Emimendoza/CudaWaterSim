#include "window.h"
#include "../utils.h"
#include "bgfx/bgfx.h"
#include "bgfx/platform.h"
#include "platform_specific.h"

using namespace waterSim::utils;

namespace waterSim::ui {
    window::window() : window(800, 600) {}

    window::window(int h, int w) {
        if(SDL_Init(SDL_INIT_VIDEO) < 0){
            printE("Failed to initialize SDL: {}\n", SDL_GetError());
            throw std::runtime_error("Failed to initialize SDL");
        }
        height = h;
        width = w;
        print("SDL initialized.\n");
        windowPtr = SDL_CreateWindow("Water Simulation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
        if (windowPtr == nullptr){
            printE("Failed to create window: {}\n", SDL_GetError());
            throw std::runtime_error("Failed to create window");
        }
        print("Window created.\n");
        running = true;
        SDL_SysWMinfo info;
        SDL_VERSION(&info.version)
        if (!SDL_GetWindowWMInfo(windowPtr, &info)){
            printE("Failed to get window info: {}\n", SDL_GetError());
            throw std::runtime_error("Failed to get window info");
        }
        print("Window info retrieved.\n");

        bgfx::PlatformData pd;
        initPd(pd, info);
        bgfx::renderFrame();
        bgfx::Init init;
        init.platformData = pd;

        if(!bgfx::init(init)){
            printE("Failed to initialize bgfx.\n");
            throw std::runtime_error("Failed to initialize bgfx");
        }
        bgfx::reset(width, height, BGFX_RESET_VSYNC);
        printV("bgfx initialized.\n");
        if (isVerbose()){
            bgfx::setDebug(BGFX_DEBUG_TEXT);
        }

        bgfx::setViewRect(0, 0, 0, width, height);
        bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x000000ff, 1.0f, 0);
        bgfx::touch(0);

    }

    window::~window() {
        if(running){
            running = false;
        }
        SDL_DestroyWindow(windowPtr);
        bgfx::shutdown();
        SDL_Quit();
    }

    void window::mainLoop() {
        SDL_Event event;
        while(running){
            bgfx::frame();
            if(SDL_PollEvent(&event) != 0){
                handleEvent(event);
            }
        }
    }

    void window::run() {
        mainLoop();
    }

    void window::handleEvent(const SDL_Event &event) {
        switch(event.type){
            case SDL_QUIT:
                running = false;
                printS("SDL_QUIT event received.\n");
                break;
            default:
                printES("Unhandled event: {}\n", event.type);
                break;
        }
    }
}