#pragma once
#include "../connect4.hpp"
#include "../enums.hpp"

#include <random>
#include <string>
#include <vector>

template<int WIDTH, int HEIGHT>
struct C4RandomAgent {
    std::string name;
    std::mt19937 rng;

    C4RandomAgent() : name("Unnamed C4RandomAgent"), rng(std::random_device{}()) {}
    C4RandomAgent(std::string name) : name(std::move(name)), rng(std::random_device{}()) {}
    C4RandomAgent(std::string name, unsigned int seed) : name(std::move(name)), rng(seed) {}

    float get_eval(Connect4<WIDTH, HEIGHT>& game) {
        return 0.0f;
    }

    int get_move(Connect4<WIDTH, HEIGHT>& game) {
        std::vector<int> moves;
        for (int col = 0; col < WIDTH; col++) {
            if (game.can_play(col)) moves.push_back(col);
        }
        std::uniform_int_distribution<> distrib(0, static_cast<int>(moves.size()) - 1);
        return moves.at(distrib(rng));
    }

    std::string& get_name() {
        return name;
    }
};
