#pragma once
#include "../TicTacToe.hpp"
#include "../enums.hpp"
#include "../TranspositionTable.hpp"
#include "../../FFNN.hpp"

#include <concepts>
#include <array>
#include <random>
#include <exception>
#include <iostream>
#include <string>
#include <chrono>

template<int N, int W, int S = 0>
struct RandomAgent {
    std::string name;
    RandomAgent() : name("Unnamed RandomAgent") {}
    RandomAgent(std::string name) : name(name) {}

    float get_eval(TicTacToe<N, W>& game) {
        return 0.0;
    }

    int get_move(TicTacToe<N, W>& game) {
        std::vector<int> moves;
        for (int i = 0; i < N * N; i++) {
            if (game.at(i) != BoardSquare::EMPTY) continue;
            moves.push_back(i);
        }

        std::mt19937 mt(S);
        std::uniform_int_distribution<> distrib(0, moves.size() - 1);
        return moves[distrib(mt)];
    }

    std::string& get_name() {
        return name;
    }
};