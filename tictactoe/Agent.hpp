#pragma once
#include "./TicTacToe.hpp"
#include "./enums.hpp"
#include "./TranspositionTable.hpp"
#include "../FFNN.hpp"

#include <concepts>
#include <array>
#include <random>
#include <exception>
#include <iostream>
#include <string>
#include <chrono>

template<typename T, int N, int W>
concept Agent = requires(T agent, TicTacToe<N, W>& game) {
    { agent.get_eval(game) } -> std::convertible_to<float>;
    { agent.get_move(game) } -> std::convertible_to<int>;
    { agent.get_name() } -> std::convertible_to<std::string>;
};

#include "./agents/FFNN.hpp"
#include "./agents/Human.hpp"
#include "./agents/MinimaxRev1.hpp"
#include "./agents/MinimaxRev2.hpp"
#include "./agents/Random.hpp"