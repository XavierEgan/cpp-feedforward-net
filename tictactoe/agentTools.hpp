#pragma once
#include "./Agent.hpp"
#include "./tictactoe.hpp"
#include "./enums.hpp"
#include "../Eigen/Dense"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <ratio>
#include <stdexcept>
#include <string>

template <int N, int W, Agent<N, W> A1, Agent<N, W> A2>
void benchmark_agents(A1 a1, A2 a2, int num_tests = 100) {
    std::chrono::time_point<std::chrono::steady_clock> overall_start_time = std::chrono::steady_clock::now();
    TicTacToe<N, W> game;

    int a1_wins = 0;
    int a2_wins = 0;
    int draws = 0;

    std::chrono::duration total_time_a1 = std::chrono::duration<double, std::nano>(0);
    std::chrono::duration total_time_a2 = std::chrono::duration<double, std::nano>(0);

    int total_moves_a1 = 0;
    int total_moves_a2 = 0;

    BoardSquare a1_piece = BoardSquare::X;

    for (int test = 0; test < num_tests; test++) {
        game.restart();

        a1_piece = a1_piece == BoardSquare::X ? BoardSquare::O : BoardSquare::X;

        BoardSquare winner = BoardSquare::EMPTY;

        for (int i = 0; i < N * N; i++) {
            int move;
            const bool is_a1_turn = game.next_player == a1_piece;

            std::chrono::time_point start_time = std::chrono::steady_clock::now();
            if (is_a1_turn) move = a1.get_move(game);
            else move = a2.get_move(game);
            std::chrono::time_point end_time = std::chrono::steady_clock::now();
            
            if (is_a1_turn) {
                total_time_a1 += (end_time - start_time);
                total_moves_a1++;
            } else {    
                total_time_a2 += (end_time - start_time);
                total_moves_a2++;
            }

            if (!game.play_move(move)) throw std::runtime_error("agent played invalid move");
            winner = game.check_winner(move);
            if (winner != BoardSquare::EMPTY) break;

        }

        if (winner == BoardSquare::EMPTY) draws++;
        else if (winner == a1_piece) a1_wins++;
        else a2_wins++;
        
        std::cout << "game " << test + 1 << " played" << std::endl;

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

    std::chrono::duration<double, std::milli> avg_move_time_a1 = total_time_a1 / total_moves_a1;
    std::chrono::duration<double, std::milli> avg_move_time_a2 = total_time_a2 / total_moves_a2;

    std::cout << a1.get_name() << " avg move time: " << avg_move_time_a1.count() << std::endl;
    std::cout << a2.get_name() << " avg move time: " << avg_move_time_a2.count() << std::endl;

    std::chrono::time_point<std::chrono::steady_clock> overall_end_time = std::chrono::steady_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time).count() << "ms" << std::endl;
}

template <int N, int W, Agent<N, W> A1, Agent<N, W> A2>
/*
inputs gets board states appended to it
targets gets board state evaluation from a1 appended to it
*/
void get_training_data(A1 a1, A2 a2, std::vector<Eigen::MatrixXf>& inputs, std::vector<Eigen::MatrixXf>& targets, int num_games = 1000) {
    TicTacToe<N, W> game;

    BoardSquare a1_piece = BoardSquare::X;

    for (int g = 0; g < num_games; g++) {
        game.restart();

        a1_piece = a1_piece == BoardSquare::X ? BoardSquare::O : BoardSquare::X;

        std::cout << "game " << g + 1 << "/" << num_games << " played" << std::endl;

        for (int i = 0; i < N * N; i++) {
            int move;
            if (game.next_player == a1_piece) move = a1.get_move(game);
            else move = a2.get_move(game);
            game.play_move(move);

            inputs.push_back(game.get_board_state());
            Eigen::MatrixXf target(1, 1);
            target(0, 0) = a1.get_eval(game);
            if (game.next_player == BoardSquare::O) target *= -1;
            targets.push_back(target);

            BoardSquare winner = game.check_winner(move);
            if (winner != BoardSquare::EMPTY) break;
        }
    }
}