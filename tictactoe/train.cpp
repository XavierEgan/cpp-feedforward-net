// g++ -std=c++23 -O3 -o train.exe tictactoe/train.cpp && train.exe

#include "../AdamOptimiser.hpp"
#include "../NN_Utils.hpp"
#include "./agentTools.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <filesystem>

#include <immintrin.h>

// train a model to copy minimax_agent to get started
void train_base_model() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    const int board_size = 5;
    const int win_length = 4;

    // gather data for pretraining
    MinimaxAgent<board_size, win_length> minimax_agent(3);

    std::vector<Eigen::MatrixXf> inputs;
    std::vector<Eigen::MatrixXf> targets;
    get_training_data<board_size, win_length>(minimax_agent, minimax_agent, inputs, targets);

    // train on the data
    std::vector<size_t> network_shape = {board_size * board_size, 64, 32, 1};
    std::vector<ActivationFunc> activation_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::tan_h};

    FFNN ffnn = FFNN::from_random_he_scaling(network_shape, activation_funcs);
    AdamOptimiser optimiser = AdamOptimiser(ffnn, CostType::quadratic);

    Eigen::MatrixXf minibatch;
    Eigen::MatrixXf minibatch_targets;
    nn_utils::get_random_batch(inputs, targets, minibatch, minibatch_targets, 200);

    const int num_epochs = 10000;

    for (int i = 0; i < num_epochs; i++) {
        optimiser.step(minibatch, minibatch_targets);
        std::cout << "Epoch: " << i + 1 << " done" << std::endl;
    }
}

int main() {
    train_base_model();
    return 0;
}