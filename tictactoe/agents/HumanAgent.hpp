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
struct HumanAgent {
    std::string name;

    HumanAgent() : name("Unnamed HuamnAgent") {}
    HumanAgent(std::string name) : name(name) {}

    float get_eval(TicTacToe<N, W>& game) {
        float user_eval;
        do {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "If -1 is winning for O and 1 is winning for X, how would evaluate the game?" << std::endl;
        } while (!(std::cin >> user_eval));

        return user_eval;
    }

    int get_move(TicTacToe<N, W>& game) {
        game.print_board();

        int move_x;
        int move_y;

        while (true) {
            std::cout << "Move x: ";
            while (!(std::cin >> move_x)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid number. Move x: ";
            }

            std::cout << "Move y: ";
            while (!(std::cin >> move_y)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid number. Move y: ";
            }

            if (move_x < 0 || move_x >= N || move_y < 0 || move_y >= N) std::cout << "Invalid move" << std::endl;
            else if (game.at(move_x, move_y) != BoardSquare::EMPTY)  std::cout << "Invalid move" << std::endl;
            else break;
        }

        return move_y * N + move_x;
    }

    std::string get_name() {
        return name;
    }
};