#pragma once
#include "./connect4.hpp"
#include "./enums.hpp"

#include <concepts>
#include <string>

/*
agents mirror the tictactoe Agent concept:
    get_eval(game) – evaluation of the current position from RED's perspective
                     (+1 winning for red, -1 winning for yellow)
    get_move(game) – column to play (0 .. WIDTH-1)
    get_name()     – display name for benchmarks
*/
template<typename T, int WIDTH, int HEIGHT>
concept C4Agent = requires(T agent, Connect4<WIDTH, HEIGHT>& game) {
    { agent.get_eval(game) } -> std::convertible_to<float>;
    { agent.get_move(game) } -> std::convertible_to<int>;
    { agent.get_name() } -> std::convertible_to<std::string>;
};

#include "./agents/FFNN.hpp"
#include "./agents/Human.hpp"
#include "./agents/Negamax.hpp"
#include "./agents/Random.hpp"
