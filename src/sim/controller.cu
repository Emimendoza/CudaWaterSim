#include "controller.cuh"

waterSim::sim::controller::controller(size_t pointCount, float radius) {
    this -> pointCount = pointCount;
    cudaMallocHost(&pointsHost, sizeof(point) * pointCount);
    pointsPosHost = new vec3*[pointCount];
    cudaMalloc(&pointsDevice, sizeof(point) * pointCount);
    modifierCount = 0;
    modifiersHost = nullptr;
    modifiersDevice = nullptr;
    for (size_t i = 0; i < pointCount; i++){
        pointsHost[i] = point(radius);
        pointsPosHost[i] = &pointsHost[i].pos;
    }
    for (size_t i = 0; i < pointCount; i++) {
        pointsPosHost[i] = &pointsHost[i].pos;
    }
}
