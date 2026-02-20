// g++ -std=c++23 -O3 -o play.exe tictactoe/interface.cpp && play.exe

#include "tictactoe.hpp"
#include "../FFNN.hpp"
#include "./Agent.hpp"
#include "./agentTools.hpp"

#include <iostream>
#include <time.h>

void play() {
    const int board_size = 3;
    const int win_length = 3;

    MinimaxAgent<board_size, win_length> mini_shallow(6);
    HumanAgent<board_size, win_length> human;

    benchmark_agents<board_size, win_length>(mini_shallow, human);
}

int main() {
    play();
    return 0;
}