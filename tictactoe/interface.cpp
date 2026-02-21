// g++ -std=c++23 -O3 -o play.exe tictactoe/interface.cpp && play.exe

#include "tictactoe.hpp"
#include "../FFNN.hpp"
#include "./Agent.hpp"
#include "./agentTools.hpp"

#include <iostream>
#include <time.h>

void play() {
    const int board_size = 4;
    const int win_length = 4;

    MinimaxAgent<board_size, win_length> mini_shallow;
    HumanAgent<board_size, win_length> human;
    RandomAgent<board_size, win_length> random;

    benchmark_agents<board_size, win_length>(mini_shallow, random, 1);
}

int main() {
    play();
    return 0;
}