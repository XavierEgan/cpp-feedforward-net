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

    MinimaxRev1Agent<board_size, win_length> mini_rev1(5);
    MinimaxRev2Agent<board_size, win_length> mini_rev2(4);
    MinimaxRev3Agent<board_size, win_length> mini_rev3(100);
    MinimaxRev4Agent<board_size, win_length> mini_rev4(100);
    HumanAgent<board_size, win_length> human;
    RandomAgent<board_size, win_length> random;

    benchmark_agents<board_size, win_length>(mini_rev4, mini_rev3, 50);
    
}

int main() {
    play();
    return 0;
}