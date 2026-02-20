#pragma once
#include "./Agent.hpp"
#include "./TicTacToe.hpp"
#include "./enums.hpp"

#include <iostream>
#include <iomanip>

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
            std::cout << "a" << std::endl;
            if (game.next_player == BoardSquare::X) move = ax.get_move(game);
            std::cout << "b" << std::endl;
            if (game.next_player == BoardSquare::O) move = ao.get_move(game);
            std::cout << "c" << std::endl;
            game.play_move(move);
            winner = game.check_winner(move);
            if (winner != BoardSquare::EMPTY) break;

        }

        if (winner == BoardSquare::X) x_wins++;
        if (winner == BoardSquare::O) o_wins++;
        if (winner == BoardSquare::EMPTY) draws++;
    }

    std::cout << " x wins: " << x_wins << "  o wins: " << o_wins << "  draws: " << draws << std::endl;
}