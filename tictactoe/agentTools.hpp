#pragma once
#include "./Agent.hpp"
#include "./TicTacToe.hpp"
#include "./enums.hpp"

#include <iostream>
#include <iomanip>
#include <string>

template <int N, int W, Agent<N, W> AX, Agent<N, W> AO>
void benchmark_agents(AX ax, AO ao, int num_tests = 100) {
    TicTacToe<N, W> game;

    int x_wins = 0;
    int o_wins = 0;
    int draws = 0;

    for (int test = 0; test < num_tests; test++) {
        game.restart();

        BoardSquare winner;

        for (int i = 0; i < N * N; i++) {
            int move;
            if (game.next_player == BoardSquare::X) move = ax.get_move(game);
            if (game.next_player == BoardSquare::O) move = ao.get_move(game);
            game.play_move(move);
            winner = game.check_winner(move);
            if (winner != BoardSquare::EMPTY) break;

        }

        if (winner == BoardSquare::X) x_wins++;
        if (winner == BoardSquare::O) o_wins++;
        if (winner == BoardSquare::EMPTY) draws++;
    }

    float x_ratio = static_cast<float>(x_wins) / num_tests;
    float draw_ratio = static_cast<float>(draws) / num_tests;
    float o_ratio = static_cast<float>(o_wins) / num_tests;

    const int bar_width = 50;
    const std::string x_color = "\033[32m";
    const std::string draw_color = "\033[37m";
    const std::string o_color = "\033[31m";
    const std::string reset = "\033[0m";
    
    std::cout << '[';

    for (int i = 0; i < bar_width; i++) {
        float ratio = static_cast<float>(i) / bar_width;
        if (ratio < x_ratio) std::cout << x_color;
        else if (ratio < x_ratio + draw_ratio) std::cout << draw_color;
        else std::cout << o_color;
        std::cout << '=';
    }
    std::cout << reset << ']';

    std::cout << " " << x_color << ax.get_name() << ": " << x_wins << " wins";
    std::cout << " " << o_color << ao.get_name() << ": " << o_wins << " wins";
    std::cout << " " << draw_color << "draws: " << draws << reset << std::endl;
}

