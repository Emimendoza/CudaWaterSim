#pragma once
#include <bgfx/bgfx.h>
#include <SDL2/SDL_syswm.h>

void initPd(bgfx::PlatformData& pd, SDL_SysWMinfo& info){
    #if __gnu_linux__ || __OpenBSD__ || __FreeBSD__
        pd.ndt = info.info.x11.display;
        pd.nwh = (void*)(uintptr_t)info.info.x11.window;
    #elif __WIN32__
        pd.ndt = nullptr;
        pd.nwh = info.info.win.window;
    #else
    #error "Unsupported platform"
    #endif // BX_PLATFORM_
}