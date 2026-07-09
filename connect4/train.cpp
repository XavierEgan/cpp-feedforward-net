// g++ -std=c++23 -O3 -o c4_train connect4/train.cpp && ./c4_train
//
// full training pipeline (run from the repo root):
//   1. generate self-play games between negamax agents with epsilon exploration
//   2. train an FFNN value net to predict the game outcome from the board
//   3. benchmark the net (as a search evaluator) against random and negamax
//
// usage: ./build/c4_train [num_games] [teacher_ms] [train_steps]
// defaults are small enough for a first run in a few minutes; crank num_games
// and teacher_ms for a stronger net

#include "./connect4.hpp"
#include "./Agent.hpp"
#include "./agentTools.hpp"
#include "../FFNN.hpp"
#include "../AdamOptimiser.hpp"
#include "../Trainer.hpp"
#include "../DataSet.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <future>
#include <chrono>

constexpr int width = 7;
constexpr int height = 6;

// mse of the net's outcome predictions on a held-out set
float value_mse(const FFNN& ffnn, const DataSet& data) {
    const Eigen::MatrixXf predictions = ffnn.forward(data.inputs);
    return (predictions - data.labels).squaredNorm() / static_cast<float>(data.size());
}

int main(int argc, const char* argv[]) {
    const size_t train_steps = (argc > 3) ? static_cast<size_t>(std::stoul(argv[3])) : 20000;

    // ── 1. get training data from pre computed runs ─────────────────────
    // get all files that start with "training_data_w7_h6_" and end with ".bin" in the connect4 folder
    std::vector<std::string> training_files;
    for (const auto& entry : std::filesystem::directory_iterator("connect4")) {
        const std::string filename = entry.path().filename().string();
        if (filename.rfind("training_data_w7_h6_", 0) == 0 && filename.size() > 4 && filename.substr(filename.size() - 4) == ".bin") {
            training_files.push_back(entry.path().string());
        }
    }

    if (training_files.empty()) {
        throw std::runtime_error("no training data files found in connect4/; run generate_training_data first");
    }

    DataSet all_data = DataSet::from_files(training_files);
    
    // 90/10 train/test split (games are appended in order, so take the tail as test)
    const Eigen::Index n = all_data.inputs.cols();
    const Eigen::Index n_train = n * 9 / 10;

    DataSet train_data = DataSet::from_matrices(all_data.inputs.leftCols(n_train), all_data.labels.leftCols(n_train));
    DataSet test_data = DataSet::from_matrices(all_data.inputs.rightCols(n - n_train), all_data.labels.rightCols(n - n_train));

    // ── 2. train the value net ────────────────────────────────────────────────
    // 42 board cells in, single tanh outcome prediction out
    std::vector<size_t> shape = {width * height, 128, 64, 1};
    std::vector<ActivationFunc> acts = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::tan_h};

    FFNN model = FFNN::from_random_he_scaling(shape, acts);
    // l2 keeps the net from memorising the (noisy) outcome labels
    AdamOptimiser opt = AdamOptimiser::from_ffnn(model, CostType::mse, 1e-3f, RegularizationType::l2, 1e-3f);

    std::filesystem::create_directories("connect4/models");

    TrainSettings settings;
    settings.num_steps = train_steps;
    settings.batch_size = 256;
    settings.eval_interval = 500;
    settings.print_interval = 1000;
    settings.checkpoint_path = "connect4/models/best.dat";

    std::cout << "\ntraining on " << train_data.size() << " positions, testing on " << test_data.size() << "..." << std::endl;

    // eval is negated mse so that higher = better for checkpointing
    const TrainResult result = train(model, opt, train_data, settings,
        [&](const FFNN& m) { return -value_mse(m, test_data); });

    std::cout << "best test mse: " << -result.best_score << " at step " << result.best_step << std::endl;

    // ── 3. benchmark the trained net as a search evaluator ────────────────────
    FFNN best = FFNN::from_file("connect4/models/best.dat");

    C4FFNNAgent<width, height> ffnn_agent(10.0, "ffnn-10ms", best);
    C4RandomAgent<width, height> random("random");
    C4NegamaxAgent<width, height> weak_negamax(1.0, "negamax-1ms");

    std::cout << "\n=== ffnn vs random ===" << std::endl;
    c4_benchmark_agents<width, height>(ffnn_agent, random, 100, true);

    std::cout << "\n=== ffnn vs negamax-1ms ===" << std::endl;
    c4_benchmark_agents<width, height>(ffnn_agent, weak_negamax, 50, true);

    return 0;
}
