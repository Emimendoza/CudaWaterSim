#include "file.h"
#include "../math.h"
#include "../../utils.h"

#pragma clang diagnostic push // Ignore warnings caused by isLittleEndian
#pragma ide diagnostic ignored "UnreachableCode"
#pragma ide diagnostic ignored "ConstantConditionsOC"
#pragma ide diagnostic ignored "Simplify"

struct bitHolder{
    unsigned char bits;
    unsigned char bitCount;
};

bool encodeBit(const bool &b, bitHolder& holder, bool flush, std::ofstream& file){
    holder.bits <<= 1;
    if (b){
        holder.bits |= 1;
    }
    holder.bitCount++;
    if (flush){
        for (int i = 0; i < 8 - holder.bitCount; i++){
            holder.bits <<= 1;
            holder.bitCount++;
        }
    }
    if (holder.bitCount == 8){
        file << holder.bits;
        holder.bits = 0;
        holder.bitCount = 0;
        return true;
    }
    return false;
}
bool waterSim::sim::file::writeBakedPoints(waterSim::sim::bakedPoint *points, const size_t &pointCount,
                                           const std::string &path) {
    std::ofstream file = std::ofstream(path, std::ios::binary);
    if (!file.is_open()){
        utils::printES("Failed to open file for writing.\n");
        return false;
    }
    file << static_cast<unsigned char>(0); // Version 0
    return encodeBakedPointsVersion0(points, pointCount, file);
}

float waterSim::sim::file::floatToLittleEndian(float f){
    if(isLittleEndian()){
        return f;
    } else {
        auto *bytes = reinterpret_cast<unsigned char*>(&f);
        unsigned char temp = bytes[0];
        bytes[0] = bytes[3];
        bytes[3] = temp;
        temp = bytes[1];
        bytes[1] = bytes[2];
        bytes[2] = temp;
        return *reinterpret_cast<float*>(bytes);
    }
}

double waterSim::sim::file::doubleToLittleEndian(double d) {
    if(isLittleEndian()){
        return d;
    } else {
        auto *bytes = reinterpret_cast<unsigned char*>(&d);
        unsigned char temp = bytes[0];
        bytes[0] = bytes[7];
        bytes[7] = temp;
        temp = bytes[1];
        bytes[1] = bytes[6];
        bytes[6] = temp;
        temp = bytes[2];
        bytes[2] = bytes[5];
        bytes[5] = temp;
        temp = bytes[3];
        bytes[3] = bytes[4];
        bytes[4] = temp;
        return *reinterpret_cast<double*>(bytes);
    }
}

unsigned long waterSim::sim::file::ulToLittleEndian(unsigned long ul) {
    if (isLittleEndian()) {
        return ul;
    } else {
        auto *bytes = reinterpret_cast<unsigned char *>(&ul);
        unsigned char temp = bytes[0];
        bytes[0] = bytes[7];
        bytes[7] = temp;
        temp = bytes[1];
        bytes[1] = bytes[6];
        bytes[6] = temp;
        temp = bytes[2];
        bytes[2] = bytes[5];
        bytes[5] = temp;
        temp = bytes[3];
        bytes[3] = bytes[4];
        bytes[4] = temp;
        return *reinterpret_cast<unsigned long *>(bytes);
    }
}
#pragma clang diagnostic pop