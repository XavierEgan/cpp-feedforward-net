// g++ -std=c++23 -O3 -o train.exe tictactoe/train.cpp && train.exe

#include "../AdamOptimiser.hpp"
#include "../NN_Utils.hpp"
#include "../DataSet.hpp"
#include "./agentTools.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>



int main(int argc, const char* argv[]) {
    // get training data
    
    
    DataSet training_data = get_training_data();


}