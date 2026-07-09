#pragma once

// scoped enums so these never collide with tictactoe's BoardSquare/NodeType
// if both games end up in one translation unit

enum class Disc {
    RED = 0,
    YELLOW = 1,
    EMPTY = 2,
};

enum class Bound {
    EXACT,
    LOWER,
    UPPER,
};

inline char disc_to_char(Disc disc) {
    if (disc == Disc::RED) {
        return 'R';
    } else if (disc == Disc::YELLOW) {
        return 'Y';
    } else {
        return ' ';
    }
}

inline Disc other_player(Disc disc) {
    return (disc == Disc::RED) ? Disc::YELLOW : Disc::RED;
}
