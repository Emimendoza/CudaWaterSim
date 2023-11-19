#include "controller.cuh"
#include "sim.cuh"
#include <algorithm>

waterSim::sim::controller::controller(size_t pointCount, float radius) {
    this -> pointCount = pointCount;
    pointsPosHostActive = new vec3[pointCount];
    cudaMallocHost(&pointsHost, sizeof(point) * pointCount);
    cudaMalloc(&pointsDevice, sizeof(point) * pointCount);
    for (size_t i = 0; i < pointCount; i++){
        pointsHost[i] = point(radius);
    }
}

waterSim::sim::controller::~controller() {
    delete[] pointsPosHostActive;
    cudaFreeHost(pointsHost);
    cudaFree(pointsDevice);
    cudaFree(modifiersDevice);
}

void waterSim::sim::controller::addModifier(waterSim::sim::modifierI *m) {
    modifiersHost.push_back(m);
    if (modifierArraySize < modifiersHost.size()){
        modifierArraySize = modifiersHost.size()*2;
        cudaFree(modifiersDevice);
        cudaMalloc(&modifiersDevice, sizeof(modifierI*) * modifierArraySize);
    }
    cudaMemcpy(modifiersDevice, modifiersHost.data(), sizeof(modifierI*) * modifiersHost.size(), cudaMemcpyHostToDevice);
}

void waterSim::sim::controller::removeModifier(waterSim::sim::modifierI *m) {
    modifiersHost.erase(std::remove(modifiersHost.begin(), modifiersHost.end(), m), modifiersHost.end());
    cudaMemcpy(modifiersDevice, modifiersHost.data(), sizeof(modifierI*) * modifiersHost.size(), cudaMemcpyHostToDevice);
}

void waterSim::sim::controller::syncDeviceToHost() {
    cudaMemcpy(pointsHost, pointsDevice, sizeof(point) * pointCount, cudaMemcpyDeviceToHost);
}

void waterSim::sim::controller::syncHostToDevice() {
    cudaMemcpy(pointsDevice, pointsHost, sizeof(point) * pointCount, cudaMemcpyHostToDevice);
}

void waterSim::sim::controller::step() {
    runModifiers();
    runGravity();
    runCollision();
    updateGraphics();
}

[[noreturn]] void waterSim::sim::controller::mainLoop() {
    while (true){ // TODO: add exit condition
        step();
    }
}

void waterSim::sim::controller::runModifiers() {
    modifyPoints<<<(pointCount + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(pointsDevice, modifiersDevice, modifiersHost.size(), pointCount);
}