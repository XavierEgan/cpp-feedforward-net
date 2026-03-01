// g++ -std=c++23 -O3 -o play.exe tictactoe/interface.cpp && play.exe

#include "tictactoe.hpp"
#include "../FFNN.hpp"
#include "./Agent.hpp"
#include "./agentTools.hpp"

#include <iostream>
#include <time.h>

void play() {
    const int board_size = 5;
    const int win_length = 4;

    OldMinimaxAgent<board_size, win_length> old_mini(5, "old MiniMaxAgent");
    MinimaxAgent<board_size, win_length> new_mini(3, "new MiniMaxAgent");
    HumanAgent<board_size, win_length> human;
    RandomAgent<board_size, win_length> random;

    benchmark_agents<board_size, win_length>(old_mini, new_mini);
}

int main() {
    play();
    return 0;
}