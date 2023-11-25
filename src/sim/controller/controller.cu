#include "controller.cuh"
#include "../sim.cuh"
#include "../../utils.h"
#include <algorithm>
#include "../constants.h"

waterSim::sim::controller::controller(size_t pointCount, float radius, vec3 domainSize) : simulationDomain({}, domainSize, {}) {
    this -> pointCount = pointCount;
    pointsPosHostActive = new vec3[pointCount];
    cudaMallocHost(&pointsHost, sizeof(point) * pointCount);
    cudaMalloc(&pointsDevice, sizeof(point) * pointCount);
    bakedPointsBuffer = new bakedPoint[pointCount];
    for (size_t i = 0; i < pointCount; i++){
        pointsHost[i] = point(radius);
    }
}

waterSim::sim::controller::~controller() {
    if(simRunning){
        utils::printES("Simulator was running when it was destroyed. Waiting for it to finish.\n");
        simThread.join();
    }else if (simStarted){
        // A thread must be joined or detached this is the case where the simulation is done
        simThread.join();
    }
    delete[] bakedPointsBuffer;
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
    runCollision();
    updateGraphics();
    if(baking and bakedFrameCount%bakedFrameSkip == 0){
        bakeFrame();
    }
}

void waterSim::sim::controller::mainLoop(std::atomic<bool> &condition) {
    simRunning = true;
    while (condition){
        if(!simPaused){
            step();
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(TIME_OUT_MS));
    }
    simRunning = false;
}

void waterSim::sim::controller::runModifiers() {
    modifyPoints<<<(pointCount + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(pointsDevice, modifiersDevice, modifiersHost.size(), pointCount);
}

void waterSim::sim::controller::runCollision() {
    // TODO: implement
}

void waterSim::sim::controller::updateGraphics() {
    // TODO: implement
}

void waterSim::sim::controller::run(std::atomic<bool>& condition) {
    if(simStarted){
        utils::printES("Simulator was already started. Ignoring.\n");
        return;
    }
    simStarted = true;
    simThread = std::thread(&controller::mainLoop, this, std::ref(condition));
}

void waterSim::sim::controller::pause() {
    if (simPaused){
        utils::printES("Simulator was already paused. Ignoring.\n");
        return;
    }
    simPaused = true;
}

void waterSim::sim::controller::resume() {
    if (!simPaused){
        utils::printES("Simulator was not paused. Ignoring.\n");
        return;
    }
    simPaused = false;
}

void waterSim::sim::controller::bakeFrame() {
    auto pointer = pointersFree.pop();
    cudaMemcpy(pointer, pointsDevice, sizeof(point) * pointCount, cudaMemcpyDeviceToHost);
    pointersFilled.push(pointer);
}

void waterSim::sim::controller::bake(size_t frameCount, size_t frameSkip) {
    if(baking){
        utils::printES("Simulator was already baking. Ignoring.\n");
        return;
    }
    if(simStarted){
        utils::printES("Simulator was already started. Ignoring.\n");
        return;
    }
    for (auto & pointer : pointsBuffer){
        if(cudaMallocHost(&pointer, sizeof(point) * pointCount) != cudaSuccess){
            utils::printES("Failed to allocate memory for baking. Aborting.\n");
            return;
        }
        pointersFree.push(pointer);
    }
    baking = true;
    bakedFrameCount = frameCount;
    bakedFrameSkip = frameSkip;
}

bool waterSim::sim::controller::setBakePath(const std::string &path) {
    if (baking){
        utils::printES("Simulator was already baking. Ignoring.\n");
        return false;
    }
    bakePath = path;
    return true;
}

void waterSim::sim::controller::bakeFrameToFile() {
    auto pointer = pointersFilled.pop();
    for (size_t i = 0; i < pointCount; i++){
        bakedPointsBuffer[i] = bakedPoint(pointer[i].getPositionPrimitive(), pointer[i].active);
    }
    pointersFree.push(pointer);
    // TODO: write to file
}
