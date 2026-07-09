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

template<int WIDTH, int HEIGHT, C4Agent<WIDTH, HEIGHT> A1, C4Agent<WIDTH, HEIGHT> A2>
void c4_benchmark_agents(A1& a1, A2& a2, int num_tests = 100, bool quiet = false) {
    const std::chrono::steady_clock::time_point overall_start_time = std::chrono::steady_clock::now();
    Connect4<WIDTH, HEIGHT> game;

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
            int move;
            if (is_a1_turn) move = a1.get_move(game);
            else move = a2.get_move(game);
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
plays a1 vs a2 and records every position of every game, labelled with the
final outcome from RED's perspective (+1 red win, -1 yellow win, 0 draw).

outcome labels are simpler and less noisy than teacher-eval labels: the net
learns "who ends up winning from here", which is exactly what a value net
inside search needs.

epsilon is the chance each move is replaced by a uniform random legal move —
without it, deterministic agents replay near-identical games and the dataset
collapses to a few hundred distinct positions.

gamma discounts the label by distance from the end of the game: the final
position gets the full ±1, earlier positions shrink toward 0. early-game
outcomes are close to coin flips (any epsilon move can swing them), so an
undiscounted ±1 label there is mostly noise the net would try to memorise
*/
template<int WIDTH, int HEIGHT, C4Agent<WIDTH, HEIGHT> A1, C4Agent<WIDTH, HEIGHT> A2>
DataSet c4_get_training_data(A1& a1, A2& a2, int num_games = 1000, float epsilon = 0.1f,
                             float gamma = 0.95f, unsigned int seed = std::random_device{}(),
                             bool quiet = true) {
    Connect4<WIDTH, HEIGHT> game;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    std::vector<Eigen::VectorXf> inputs;
    std::vector<Eigen::VectorXf> labels;

    Disc a1_piece = Disc::RED;

    for (int g = 0; g < num_games; g++) {
        game.restart();
        a1_piece = other_player(a1_piece);

        const size_t game_start = inputs.size();
        Disc winner = Disc::EMPTY;

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
            inputs.push_back(game.get_board_state());

            winner = game.check_winner();
            if (winner != Disc::EMPTY) break;
        }

        // label every position of this game with the final outcome
        float outcome = 0.0f;
        if (winner == Disc::RED) outcome = 1.0f;
        else if (winner == Disc::YELLOW) outcome = -1.0f;

        const size_t game_end = inputs.size();
        for (size_t i = game_start; i < game_end; i++) {
            const int plies_to_end = static_cast<int>(game_end - 1 - i);
            Eigen::VectorXf target(1);
            target(0) = outcome * std::pow(gamma, static_cast<float>(plies_to_end));
            labels.push_back(target);

            // connect 4 is symmetric under horizontal mirroring, so every
            // position doubles into a second free sample
            Eigen::VectorXf mirrored(WIDTH * HEIGHT);
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    mirrored(y * WIDTH + x) = inputs.at(i)(y * WIDTH + (WIDTH - 1 - x));
            inputs.push_back(mirrored);
            labels.push_back(target);
        }

        if (!quiet) std::cout << "game " << g + 1 << "/" << num_games << " played (" << inputs.size() << " positions)" << std::endl;
    }
    if (inputs.size() != labels.size()) throw std::runtime_error("c4_get_training_data: inputs and labels size mismatch");
    if (inputs.size() == 0 || labels.size() == 0) return DataSet::empty(WIDTH * HEIGHT, 1);
    return DataSet::from_samples(inputs, labels);
}
