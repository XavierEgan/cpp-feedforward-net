// g++ -std=c++23 -O3 -o train.exe tictactoe/train.cpp && train.exe

#include "../AdamOptimiser.hpp"
#include "../NN_Utils.hpp"
#include "./agentTools.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <filesystem>

#include <immintrin.h>
void train() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    const int board_size = 5;
    const int win_length = 4;

    // gather data for pretraining
    MinimaxAgent<board_size, win_length> minimax_agent(3);

    std::vector<Eigen::MatrixXf> inputs;
    std::vector<Eigen::MatrixXf> targets;
    get_training_data<board_size, win_length>(minimax_agent, minimax_agent, inputs, targets);
}

int main() {
    train();
    return 0;
}