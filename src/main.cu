#include "utils.h"
#include "ui/window.h"

using namespace waterSim::utils;
using namespace waterSim;

int main(int argc, char** argv){
    print("Starting Water Simulation.\n");
    ui::window window(800, 600);
    window.run();
    print("Water Simulation finished.\n");
    return 0;
}