#pragma once
#include "./Agent.hpp"
#include "./connect4.hpp"
#include "./enums.hpp"
#include "../DataSet.hpp"
#include "../Eigen/Dense"

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// dispatches to agent.get_move(game, seed) if the agent accepts a seed (negamax/ffnn use it to
// break ties), otherwise falls back to agent.get_move(game) (e.g. C4RandomAgent seeds its own rng
// at construction instead)
template<typename Agent, int WIDTH, int HEIGHT>
int c4_get_move(Agent& agent, Connect4<WIDTH, HEIGHT>& game, unsigned int seed) {
    if constexpr (requires { agent.get_move(game, seed); }) {
        return agent.get_move(game, seed);
    } else {
        return agent.get_move(game);
    }
}

template<int WIDTH, int HEIGHT, C4Agent<WIDTH, HEIGHT> A1, C4Agent<WIDTH, HEIGHT> A2>
void c4_benchmark_agents(A1& a1, A2& a2, int num_tests = 100, bool quiet = false,
                          unsigned int seed = std::random_device{}()) {
    const std::chrono::steady_clock::time_point overall_start_time = std::chrono::steady_clock::now();
    Connect4<WIDTH, HEIGHT> game;
    std::mt19937 rng(seed);

    int a1_wins = 0;
    int a2_wins = 0;
    int draws = 0;

    std::chrono::duration<double, std::nano> total_time_a1(0);
    std::chrono::duration<double, std::nano> total_time_a2(0);

    int total_moves_a1 = 0;
    int total_moves_a2 = 0;

    long long total_depth_a1 = 0;
    long long total_depth_a2 = 0;

    Disc a1_piece = Disc::RED;

    for (int test = 0; test < num_tests; test++) {
        game.restart();

        // alternate who moves first
        a1_piece = other_player(a1_piece);

        Disc winner = Disc::EMPTY;

        while (!game.is_full()) {
            const bool is_a1_turn = game.next_player() == a1_piece;

            const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
            const int move = is_a1_turn ? c4_get_move<A1, WIDTH, HEIGHT>(a1, game, rng())
                : c4_get_move<A2, WIDTH, HEIGHT>(a2, game, rng());
            const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

            if (is_a1_turn) {
                total_time_a1 += (end_time - start_time);
                total_moves_a1++;
                if constexpr (requires { a1.get_last_depth(); })
                    if (a1.get_last_depth() >= 0) total_depth_a1 += a1.get_last_depth();
            } else {
                total_time_a2 += (end_time - start_time);
                total_moves_a2++;
                if constexpr (requires { a2.get_last_depth(); })
                    if (a2.get_last_depth() >= 0) total_depth_a2 += a2.get_last_depth();
            }

            if (!game.play_move(move)) throw std::runtime_error("c4_benchmark_agents: agent played invalid move " + std::to_string(move));
            winner = game.check_winner();
            if (winner != Disc::EMPTY) break;
        }

        if (winner == Disc::EMPTY) draws++;
        else if (winner == a1_piece) a1_wins++;
        else a2_wins++;

        if (!quiet) std::cout << "game " << test + 1 << " played" << std::endl;
    }

    const float a1_ratio = static_cast<float>(a1_wins) / num_tests;
    const float draw_ratio = static_cast<float>(draws) / num_tests;

    const int bar_width = 50;
    const std::string a1_color = "\033[32m";
    const std::string draw_color = "\033[37m";
    const std::string a2_color = "\033[31m";
    const std::string reset = "\033[0m";

    std::cout << '[';
    for (int i = 0; i < bar_width; i++) {
        const float ratio = static_cast<float>(i) / bar_width;
        if (ratio < a1_ratio) std::cout << a1_color;
        else if (ratio < a1_ratio + draw_ratio) std::cout << draw_color;
        else std::cout << a2_color;
        std::cout << '=';
    }
    std::cout << reset << ']';

    std::cout << " " << a1_color << a1.get_name() << ": " << a1_wins << " wins";
    std::cout << " " << a2_color << a2.get_name() << ": " << a2_wins << " wins";
    std::cout << " " << draw_color << "draws: " << draws << reset << std::endl;

    if (total_moves_a1 > 0) {
        const std::chrono::duration<double, std::milli> avg_move_time_a1 = total_time_a1 / total_moves_a1;
        std::cout << a1.get_name() << " avg move time: " << avg_move_time_a1.count() << "ms" << std::endl;
    }
    if (total_moves_a2 > 0) {
        const std::chrono::duration<double, std::milli> avg_move_time_a2 = total_time_a2 / total_moves_a2;
        std::cout << a2.get_name() << " avg move time: " << avg_move_time_a2.count() << "ms" << std::endl;
    }

    if constexpr (requires { a1.get_last_depth(); })
        std::cout << a1.get_name() << " avg depth: " << static_cast<float>(total_depth_a1) / total_moves_a1 << std::endl;
    if constexpr (requires { a2.get_last_depth(); })
        std::cout << a2.get_name() << " avg depth: " << static_cast<float>(total_depth_a2) / total_moves_a2 << std::endl;

    const std::chrono::steady_clock::time_point overall_end_time = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time).count() << "ms" << std::endl;
}

/*
plays a1 vs a2 and records every position of every game. non-terminal
positions are labelled with `evaluator`'s get_eval() (a fixed-depth search
score from RED's perspective) rather than the game's eventual outcome:
the final result of a single self-play game is a noisy, high-variance
sample (especially with epsilon exploration steering play afterwards), so
using it as the label for every position in the game floods training with
label noise. a per-position search score is a direct, low-variance estimate
of that position alone.

the position where the game actually ends is the exception: it's a genuine
win/loss/draw, but get_eval can't see that — its tactics only detect a win
the side to move could make *now*, not one already completed on the
previous move. so terminal positions get the exact ±1/0 outcome instead of
a (wrong) shallow-search score.

epsilon is the chance each move is replaced by a uniform random legal move —
without it, deterministic agents replay near-identical games and the dataset
collapses to a few hundred distinct positions.
*/
template<int WIDTH, int HEIGHT, C4Agent<WIDTH, HEIGHT> A1, C4Agent<WIDTH, HEIGHT> A2>
DataSet c4_get_training_data(A1& a1, A2& a2, C4NegamaxAgent<WIDTH, HEIGHT>& evaluator,
                             int num_games = 1000, float epsilon = 0.1f,
                             unsigned int seed = std::random_device{}(), bool quiet = true) {
    Connect4<WIDTH, HEIGHT> game;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    std::vector<Eigen::VectorXf> inputs;
    std::vector<Eigen::VectorXf> labels;

    Disc a1_piece = Disc::RED;

    auto record = [&](const Eigen::VectorXf& board, float label) {
        Eigen::VectorXf target(1);
        target(0) = label;
        inputs.push_back(board);
        labels.push_back(target);

        // connect 4 is symmetric under horizontal mirroring, so every
        // position doubles into a second free sample
        Eigen::VectorXf mirrored(WIDTH * HEIGHT);
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                mirrored(y * WIDTH + x) = board(y * WIDTH + (WIDTH - 1 - x));
        inputs.push_back(mirrored);
        labels.push_back(target);
    };

    for (int g = 0; g < num_games; g++) {
        game.restart();
        a1_piece = other_player(a1_piece);

        while (!game.is_full()) {
            int move;
            if (unit(rng) < epsilon) {
                std::vector<int> legal;
                for (int col = 0; col < WIDTH; col++)
                    if (game.can_play(col)) legal.push_back(col);
                std::uniform_int_distribution<int> dist(0, static_cast<int>(legal.size()) - 1);
                move = legal.at(dist(rng));
            } else if (game.next_player() == a1_piece) {
                move = a1.get_move(game);
            } else {
                move = a2.get_move(game);
            }

            game.play_move(move);

            const Disc winner = game.check_winner();
            float label;
            if (winner == Disc::RED) label = 1.0f;
            else if (winner == Disc::YELLOW) label = -1.0f;
            else if (game.is_full()) label = 0.0f; // drawn on this move
            else label = evaluator.get_eval(game);

            record(game.get_board_state(), label);

            if (winner != Disc::EMPTY) break;
        }

        if (!quiet) std::cout << "game " << g + 1 << "/" << num_games << " played (" << inputs.size() << " positions)" << std::endl;
    }
    if (inputs.size() != labels.size()) throw std::runtime_error("c4_get_training_data: inputs and labels size mismatch");
    if (inputs.size() == 0 || labels.size() == 0) return DataSet::empty(WIDTH * HEIGHT, 1);
    return DataSet::from_samples(inputs, labels);
}
