#pragma once
#include "./Agent.hpp"
#include "./TicTacToe.hpp"
#include "./enums.hpp"
#include "../Eigen/Dense"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

template <int N, int W, Agent<N, W> A1, Agent<N, W> A2>
void benchmark_agents(A1 a1, A2 a2, int num_tests = 100) {
    TicTacToe<N, W> game;
    using clock = std::chrono::steady_clock;

    int a1_wins = 0;
    int a2_wins = 0;
    int draws = 0;

    double a1_total_move_ms = 0.0;
    double a2_total_move_ms = 0.0;
    int a1_total_move_count = 0;
    int a2_total_move_count = 0;

    double a1_first_move_ms = 0.0;
    double a2_first_move_ms = 0.0;
    int a1_first_move_count = 0;
    int a2_first_move_count = 0;

    std::unordered_map<int, double> a1_move_times;
    std::unordered_map<int, double> a2_move_times;
    std::unordered_map<int, double> a1_move_counts;
    std::unordered_map<int, double> a2_move_counts;

    BoardSquare a1_piece = BoardSquare::X;

    for (int test = 0; test < num_tests; test++) {
        game.restart();

        a1_piece = a1_piece == BoardSquare::X ? BoardSquare::O : BoardSquare::X;

        BoardSquare winner = BoardSquare::EMPTY;

        for (int i = 0; i < N * N; i++) {
            int move;
            const bool is_a1_turn = game.next_player == a1_piece;
            const auto start = clock::now();
            if (is_a1_turn) move = a1.get_move(game);
            else move = a2.get_move(game);
            const auto end = clock::now();
            const double move_ms = std::chrono::duration<double, std::milli>(end - start).count();

            if (is_a1_turn) {
                a1_total_move_ms += move_ms;
                a1_total_move_count++;
                if (i == 0) {
                    a1_first_move_ms += move_ms;
                    a1_first_move_count++;
                }
                if (!a1_move_times.contains(i)) a1_move_times[i] = move_ms;
                else a1_move_times[i] += move_ms;
                if (!a1_move_counts.contains(i)) a1_move_counts[i] = 1;
                else a1_move_counts[i]++;
            } else {
                a2_total_move_ms += move_ms;
                a2_total_move_count++;
                if (i == 0) {
                    a2_first_move_ms += move_ms;
                    a2_first_move_count++;
                }
                if (!a2_move_times.contains(i)) a2_move_times[i] = move_ms;
                else a2_move_times[i] += move_ms;
                if (!a2_move_counts.contains(i)) a2_move_counts[i] = 1;
                else a2_move_counts[i]++;
            }

            game.play_move(move);
            winner = game.check_winner(move);
            if (winner != BoardSquare::EMPTY) break;

        }

        if (winner == BoardSquare::EMPTY) draws++;
        else if (winner == a1_piece) a1_wins++;
        else a2_wins++;
        
    }

    float a1_ratio = static_cast<float>(a1_wins) / num_tests;
    float draw_ratio = static_cast<float>(draws) / num_tests;

    const int bar_width = 50;
    const std::string a1_color = "\033[32m";
    const std::string draw_color = "\033[37m";
    const std::string a2_color = "\033[31m";
    const std::string reset = "\033[0m";
    
    std::cout << '[';

    for (int i = 0; i < bar_width; i++) {
        float ratio = static_cast<float>(i) / bar_width;
        if (ratio < a1_ratio) std::cout << a1_color;
        else if (ratio < a1_ratio + draw_ratio) std::cout << draw_color;
        else std::cout << a2_color;
        std::cout << '=';
    }
    std::cout << reset << ']';

    std::cout << " " << a1_color << a1.get_name() << ": " << a1_wins << " wins";
    std::cout << " " << a2_color << a2.get_name() << ": " << a2_wins << " wins";
    std::cout << " " << draw_color << "draws: " << draws << reset << std::endl;

    auto avg_or_zero = [](double total_ms, int count) {
        return count > 0 ? total_ms / static_cast<double>(count) : 0.0;
    };

    const double a1_avg_first_ms = avg_or_zero(a1_first_move_ms, a1_first_move_count);
    const double a2_avg_first_ms = avg_or_zero(a2_first_move_ms, a2_first_move_count);
    const double a1_avg_any_ms = avg_or_zero(a1_total_move_ms, a1_total_move_count);
    const double a2_avg_any_ms = avg_or_zero(a2_total_move_ms, a2_total_move_count);

    for (auto& p : a1_move_times) {
        p.second /= a1_move_counts[p.first];
    }
    for (auto& p : a2_move_times) {
        p.second /= a2_move_counts[p.first];
    }

    auto a1_longest_av_move = *std::max_element(a1_move_times.begin(), a1_move_times.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {return a.second < b.second;} );
    auto a2_longest_av_move = *std::max_element(a2_move_times.begin(), a2_move_times.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {return a.second < b.second;} );


    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nTiming summary (ms)\n";
    std::cout << std::left << std::setw(18) << "Agent"       << std::right << std::setw(16) << "First move"    << std::setw(16) << "Any move"    << std::setw(20) << "Longest Move avg" << "\n";
    std::cout << std::left << std::setw(18) << a1.get_name() << std::right << std::setw(16) << a1_avg_first_ms << std::setw(16) << a1_avg_any_ms << std::setw(10) << a1_longest_av_move.first << std::setw(10) << a1_longest_av_move.second << "\n";
    std::cout << std::left << std::setw(18) << a2.get_name() << std::right << std::setw(16) << a2_avg_first_ms << std::setw(16) << a2_avg_any_ms << std::setw(10) << a2_longest_av_move.first << std::setw(10) << a2_longest_av_move.second <<  "\n";
}


template <int N, int W, Agent<N, W> AX, Agent<N, W> AO>
/*
input gets board states appended to it
target gets board state evaluation from ax appended to it
*/
void get_training_data(AX ax, AO ao, std::vector<Eigen::MatrixXf>& inputs, std::vector<Eigen::MatrixXf>& targets, int num_games = 1000) {
    TicTacToe<N, W> game;

    for (int g = 0; g < num_games; g++) {
        game.restart();

        for (int i = 0; i < N * N; i++) {
            int move;
            if (game.next_player == BoardSquare::X) move = ax.get_move(game);
            if (game.next_player == BoardSquare::O) move = ao.get_move(game);
            game.play_move(move);

            inputs.push_back(game.get_board_state());
            Eigen::MatrixXf target(1, 1);
            target(0, 0) = ax.get_eval(game);

            BoardSquare winner = game.check_winner(move);
            if (winner != BoardSquare::EMPTY) break;
        }
    }
}