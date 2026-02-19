// g++ -std=c++23 -O3 -o train.exe tictactoe/train.cpp && train.exe

#include "tictactoe.hpp"
#include "../AdamOptimiser.hpp"
#include "../NN_Utils.hpp"
#include <vector>
#include <algorithm>
#include <iostream>

/*
play against yourself for a bit and if you win then take all the moves in the game as positive examples and vice versa
train on that and repeat
*/
#include <immintrin.h>
void naive() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    
}

int main() {
    naive();
    return 0;
}