#include <iostream>
#include <thread>
#include <future>

#include "../DataSet.hpp"
#include "connect4.hpp"
#include "Agent.hpp"
#include "agentTools.hpp"

constexpr int width = 7;
constexpr int height = 6;

// generate extremely high quality data once then reuse it for training
DataSet get_training_data(const int num_games, const double teacher_ms) {
    const int num_threads = std::thread::hardware_concurrency();
    const int games_per_thread = num_games / num_threads;
    const int remainder = num_games % num_threads;

    std::cout << "generating " << num_games << " self-play games at " << teacher_ms << "ms/move" << " with " << num_threads << " threads..." << std::endl;

    auto worker = [num_games, teacher_ms, games_per_thread, remainder](int thread_id) {
        int our_num_games = games_per_thread + (thread_id < remainder ? 1 : 0);

        if (!our_num_games) return DataSet::empty(width * height, 1);

        C4NegamaxAgent<width, height> teacher_a(teacher_ms, "teacher-a");
        C4NegamaxAgent<width, height> teacher_b(teacher_ms, "teacher-b");
        C4NegamaxAgent<width, height> evaluator;

        return c4_get_training_data<width, height>(teacher_a, teacher_b, evaluator, our_num_games, 0.15f);
    };
    std::vector<std::future<DataSet>> futures;
    for (int i = 0; i < num_threads; ++i) {
        futures.push_back(std::async(std::launch::async, worker, i));
    }

    DataSet all_data = DataSet::empty(width * height, 1);

    for (auto& fut : futures) {
        all_data.append(fut.get());
    }

    std::cout << "collected " << all_data.size() << " positions" << std::endl;
    return all_data;
}

int main(int argc, const char* argv[]) {
    const int num_games = (argc > 1) ? std::stoi(argv[1]) : 2000;
    const double teacher_ms = (argc > 2) ? std::stod(argv[2]) : 2.0;

    const DataSet training_data = get_training_data(num_games, teacher_ms);

    training_data.write_to_file("./connect4/training_data_w7_h6_" + std::to_string(static_cast<int>(teacher_ms)) + "_" + std::to_string(training_data.size()) + ".bin");
}