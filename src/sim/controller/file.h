#pragma once
#pragma clang diagnostic push // Ignore warnings caused by isLittleEndian
#pragma ide diagnostic ignored "Simplify"

#include <cstddef>
#include <string>
#include <bit>
#include "bakedPoint.h"
#include "fstream"

namespace waterSim::sim{
    // This class will handle reading and writing baked points to and from files.
    class file {
    public:
        static bool writeBakedPoints(bakedPoint *points, const size_t& pointCount, const std::string& path);

        static bakedPoint* readBakedPoints(const std::string& path, size_t& pointCount);

        static bool encodeBakedPointsVersion0(bakedPoint *points, const size_t& pointCount, std::ofstream& file);
        static bakedPoint* decodeBakedPointsVersion0(const std::string& path, size_t& pointCount);
    private:
        static bool isLittleEndian(){
            return std::endian::native == std::endian::little;
        }
        static float floatToLittleEndian(float f);
        static double doubleToLittleEndian(double d);
        static unsigned long ulToLittleEndian(unsigned long ul);
    };
}
// The file format is as follows:
// all numbers are stored in little endian (functions need to translate in case the system is big endian)
// the first byte is the file format version

// Version 0:
// the next 2 bits are whether it's a float or double (10 for float, 01 for double)
// the next 4 bytes are the number of points (this will be referenced as pointCount)
// the next pointCount bits are whether the point is active or not
// the next pointCount * 3 * sizeof(FLOAT) bytes are the positions of the points
// the last 32 bytes are the sha256 hash of the file
// all numbers are stored in little endian (this function needs to translate in case the system is big endian)

#pragma clang diagnostic pop