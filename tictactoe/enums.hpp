#pragma once

enum BoardSquare {
    X = 0,
    O = 1,
    EMPTY = 2,
};

enum NodeType {
    EXACT,
    LOWER,
    UPPER
};

char square_to_char(BoardSquare square) {
    if (square == BoardSquare::O) {
        return 'O';
    } else if (square == BoardSquare::X) {
        return 'X';
    } else {
        return ' ';
    }
}