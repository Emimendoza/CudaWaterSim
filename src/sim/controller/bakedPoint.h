#pragma once
#include "../math/vec3Primitive.h"
namespace waterSim::sim{
    struct bakedPoint{
        bakedPoint(const vec3Primitive& pos, const bool& active) : pos(pos), active(active){};
        bakedPoint() : pos({}), active(false){}
        vec3Primitive pos;
        bool active;
    };
}
