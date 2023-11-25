#include "utils.h"
#include "ui/window.h"
#include <unordered_map>

using namespace waterSim::utils;
using namespace waterSim;

constexpr auto HELP_STR = R"(
Water Simulation
Usage: waterSim [options]
Options:
    -h, --help: Print this help message.
    -v, --verbose: Print verbose output.
    --single-frame: Run a single frame of the simulation.
)";

std::unordered_map<std::string, bool> argList;

bool parseArgument(int argc, char** argv, const std::string& arg, std::string& value){
    argList[arg] = true;
    for(int i = 0; i < argc; i++){
        if(arg == argv[i]){
            if(i + 1 < argc){
                value = argv[i + 1];
                argList[value] = true;
                return true;
            }
        }
    }
    return false;
}
bool findArgument(int argc, char** argv, const std::string& arg){
    argList[arg] = true;
    for(int i = 0; i < argc; i++){
        if(arg == argv[i]){
            return true;
        }
    }
    return false;
}

void printHelp(){
    print(HELP_STR);
}

void checkForUnrecognizedArguments(int argc, char** argv){
    for(int i = 0; i < argc; i++){
        if(!argList[argv[i]]){
            printE("Unrecognized argument: {}\n", argv[i]);
            printHelp();
            exit(1);
        }
    }
}

int main(int argc, char** argv){
    argList[argv[0]] = true; // argv[0] is the program name
    if (findArgument(argc, argv, "-v") or findArgument(argc, argv, "--verbose")){
        setVerbose(true);
    }
    if (findArgument(argc, argv, "-h") or findArgument(argc, argv, "--help")){
        printHelp();
        exit(0);
    }
    bool singleFrame = findArgument(argc, argv, "--single-frame");
    checkForUnrecognizedArguments(argc, argv);
    print("Starting Water Simulation.\n");
    ui::window window(800, 600);
    if (singleFrame){
        window.singleFrame();
        print("Single frame finished.\n");
        return 0;
    }
    window.run();
    print("Water Simulation finished.\n");
    return 0;
}