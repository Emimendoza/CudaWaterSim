#include "utils.h"
#include <mutex>

#if defined __has_include && __has_include(<print>)
// Support for std::print is not yet available in GCC stdlib (as of now), but it is in clang
    #include <print>
    #define PRINT_INTERNAL std::print
    #define PRINT_INTERNAL_FMT_TYPE std::format_string<Args...>
#elif defined __has_include && __has_include(<format>)
// Support for format is in clang 14 or above or in gcc13 or above
    #include <format>
    #include <iostream>
    #define PRINT_INTERNAL std::cout << std::format
    #define PRINT_INTERNAL_FMT_TYPE std::format_string<Args...>
#else
// If using an incomplete implementation of the standard, use fmt
    #include "fmt/format.h"
    #define PRINT_INTERNAL fmt::print
    #define PRINT_INTERNAL_FMT_TYPE fmt::format_string<Args...>
#endif

namespace waterSim::utils{
    namespace{
        std::mutex printMutex;
    }

    template<typename ...Args>
    [[maybe_unused]]
    void print(PRINT_INTERNAL_FMT_TYPE fmt, Args&& ...args){
        PRINT_INTERNAL(fmt, std::forward<Args>(args)...);
    }

    template<typename ...Args>
    [[maybe_unused]]
    void printS(PRINT_INTERNAL_FMT_TYPE fmt, Args&& ...args){
        std::lock_guard<std::mutex> lock(printMutex);
        PRINT_INTERNAL(fmt, std::forward<Args>(args)...);
    }

}