#pragma once

#include <cstddef>

#if defined __has_include && __has_include(<print>)
// Support for std::print is not yet available in GCC stdlib (as of now), but it is in clang
    #include <print>
    #define PRINT_INTERNAL_FMT_TYPE std::format_string<Args...>
#elif defined __has_include && __has_include(<format>)
// Support for format is in clang 14 or above or in gcc13 or above
    #include <format>
    #define PRINT_INTERNAL_FMT_TYPE std::format_string<Args...>
#else
// If using an incomplete implementation of the standard, use fmt
    #include "fmt/format.h"
    #define PRINT_INTERNAL_FMT_TYPE fmt::format_string<Args...>
#endif


namespace waterSim::utils{
    struct transferCtx{
        void* pageLockedHostPtr1;
        void* pageLockedHostPtr2;
        size_t ptrSize;
    };

    [[maybe_unused]] bool transferCtxInit(transferCtx& ctx, size_t ptrSize);
    [[maybe_unused]] bool transferCtxDestroy(transferCtx& ctx);
    [[maybe_unused]] bool fastCudaTransferToDevice(transferCtx& ctx, void* hostPtr, void* devicePtr, size_t size);
    [[maybe_unused]] bool fastCudaTransferToHost(transferCtx& ctx, void* devicePtr, void* hostPtr, size_t size);
    template<typename ...Args>
    [[maybe_unused]] void print(PRINT_INTERNAL_FMT_TYPE fmt, Args&& ...args);
    template<typename ...Args>
    [[maybe_unused]] void printS(PRINT_INTERNAL_FMT_TYPE fmt, Args&& ...args);
}

#undef PRINT_INTERNAL_FMT_TYPE