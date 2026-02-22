#pragma once
#include "./Agent.hpp"
#include "./TicTacToe.hpp"
#include "./enums.hpp"
#include "../Eigen/Dense"

#include <iostream>
#include <iomanip>
#include <string>

template <int N, int W, Agent<N, W> A1, Agent<N, W> A2>
void benchmark_agents(A1 a1, A2 a2, int num_tests = 100) {
    TicTacToe<N, W> game;

    int a1_wins = 0;
    int a2_wins = 0;
    int draws = 0;

    BoardSquare a1_piece = BoardSquare::X;

    for (int test = 0; test < num_tests; test++) {
        game.restart();

        a1_piece = a1_piece == BoardSquare::X ? BoardSquare::X : BoardSquare::O;

        BoardSquare winner = BoardSquare::EMPTY;

        for (int i = 0; i < N * N; i++) {
            int move;
            if (game.next_player == a1_piece) move = a1.get_move(game);
            else move = a2.get_move(game);
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