// g++ -std=c++23 -O3 -o train.exe tictactoe/train.cpp && train.exe

#include "../AdamOptimiser.hpp"
#include "../NN_Utils.hpp"
#include "../TrainingData.hpp"
#include "./agentTools.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

const int board_size = 5;
const int win_length = 4;

const std::filesystem::path PREGENERATED_DATA_PATH = "tictactoe/training_runs/pregenerated_data.bin";

struct UserArgs {
    bool new_run;
    int run_index; // if not new_run
    bool use_pregenerated;
};



// returns path to the created directory
std::filesystem::path make_dir() {
    // tictactoe/training_runs
    // run.txt contains a single number which is the run number
    // /run_x
    
    // check if training_runs exists, if not create it, set run.txt to 0 and create run_0
    if (!std::filesystem::exists("tictactoe/training_runs")) {
        std::filesystem::create_directory("tictactoe/training_runs");
    }

    if (!std::filesystem::exists("tictactoe/training_runs/run.txt")) {
        std::ofstream run_file("tictactoe/training_runs/run.txt");
        run_file << "0";
        run_file.close();
        std::filesystem::create_directory("tictactoe/training_runs/run_0");
    }

    std::filesystem::path run_num_file = "tictactoe/training_runs/run.txt";
    std::ifstream run_num_file_in = std::ifstream(run_num_file);
    int run_num;
    run_num_file_in >> run_num;
    std::ofstream run_num_file_out = std::ofstream(run_num_file);
    run_num_file_out << run_num + 1;
    
    std::filesystem::path run_file = "tictactoe/training_runs/run_" + std::to_string(run_num);
    std::filesystem::create_directory(run_file);

    return run_file;
}

void store_model(std::filesystem::path path, std::string name, FFNN& ffnn) {
    ffnn.write_to_file(path / name);
}

void load_training_data(std::vector<Eigen::MatrixXf>& inputs, std::vector<Eigen::MatrixXf>& targets) {
    TrainingData td;
    td.load(PREGENERATED_DATA_PATH.string());
    inputs = std::move(td.inputs);
    targets = std::move(td.labels);
    std::cout << "Loaded " << inputs.size() << " samples from " << PREGENERATED_DATA_PATH << std::endl;
}

void save_training_data(const std::vector<Eigen::MatrixXf>& inputs, const std::vector<Eigen::MatrixXf>& targets) {
    TrainingData td;
    td.inputs = inputs;
    td.labels = targets;
    td.write(PREGENERATED_DATA_PATH.string());
    std::cout << "Saved " << inputs.size() << " samples to " << PREGENERATED_DATA_PATH << std::endl;
}

// train a model to copy minimax_agent to get started
FFNN train_base_model(UserArgs user_args) {
    std::vector<Eigen::MatrixXf> inputs;
    std::vector<Eigen::MatrixXf> targets;

    if (user_args.use_pregenerated && std::filesystem::exists(PREGENERATED_DATA_PATH)) {
        load_training_data(inputs, targets);
    } else {
        // gather data for pretraining
        MinimaxRev4Agent<board_size, win_length> minimax_agent(5);
        get_training_data<board_size, win_length>(minimax_agent, minimax_agent, inputs, targets, 1000);

        if (!std::filesystem::exists("tictactoe/training_runs"))
            std::filesystem::create_directory("tictactoe/training_runs");
        save_training_data(inputs, targets);
    }

    // train on the data
    std::vector<size_t> network_shape = {board_size * board_size, 64, 32, 1};
    std::vector<ActivationFunc> activation_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::tan_h};

    FFNN ffnn = FFNN::from_random_he_scaling(network_shape, activation_funcs);
    AdamOptimiser optimiser = AdamOptimiser(ffnn, CostType::quadratic);

    Eigen::MatrixXf minibatch;
    Eigen::MatrixXf minibatch_targets;

    const int num_epochs = 100000;

    for (int i = 0; i < num_epochs; i++) {
        nn_utils::get_random_batch(inputs, targets, minibatch, minibatch_targets, 10000);
        optimiser.step(minibatch, minibatch_targets);
        std::cout << "Epoch: " << i + 1 << " done" << std::endl;
    }

    return ffnn;
}

// returns bitpacked int with user args
UserArgs get_user_args() {
    UserArgs args;

    std::cout << "New run? (y/n): ";
    char new_run;
    std::cin >> new_run;

    if (new_run == 'y' || new_run == 'Y') {
        args.new_run = true;
        args.run_index = -1;

        std::cout << "Use pregenerated game data for first training run? (y/n): ";
        char use_pregen;
        std::cin >> use_pregen;
        args.use_pregenerated = (use_pregen == 'y' || use_pregen == 'Y');
    } else {
        args.new_run = false;
        args.use_pregenerated = false;

        std::cout << "which run to load: ";
        int run_to_load;
        std::cin >> run_to_load;
        args.run_index = run_to_load;
    }

    return args;
}

int main() {
    #if defined(__i386__) || defined(__x86_64__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    #endif

    UserArgs user_args = get_user_args();

    std::vector<size_t> network_shape = {board_size * board_size, 64, 32, 1};
    std::vector<ActivationFunc> activation_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::tan_h};

    FFNN ffnn = FFNN::from_random_he_scaling(network_shape, activation_funcs);
    std::filesystem::path path;

    if (user_args.new_run) {
        path = make_dir();
        ffnn = train_base_model(user_args);
        store_model(path, "pretrained.ffnn", ffnn);
    } else {
        path = std::filesystem::path("tictactoe/training_runs/run_" + std::to_string(user_args.run_index));
        FFNN ffnn = FFNN::from_file(path / "pretrained.ffnn");
    }

    MinimaxRev4Agent rev4 = MinimaxRev4Agent<board_size, win_length>(10);
    FFNNAgent net_agent = FFNNAgent<board_size, win_length>(10, ffnn);

    benchmark_agents<board_size, win_length>(rev4, net_agent);

    return 0;
}