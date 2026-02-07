#pragma once
#include "../Eigen/Dense"
#include <iostream>
#include <string>
#include <vector>

enum BoardSquare {
    EMPTY,
    X,
    O
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

struct TicTacToe {
    std::vector<BoardSquare> board;
    BoardSquare next_player;

    TicTacToe() {
        board = std::vector<BoardSquare>(9, BoardSquare::EMPTY);
        next_player = BoardSquare::X;
    }

    BoardSquare at(int x, int y) const {
        return board.at(y * 3 + x);
    }
    BoardSquare& at(int x, int y) {
        return board.at(y * 3 + x);
    }

    BoardSquare at(int index) const {
        return board.at(index);
    }
    BoardSquare& at(int index) {
        return board.at(index);
    }

    Eigen::MatrixXf get_board_state(BoardSquare player) {
        Eigen::MatrixXf state(9, 1);
        for (int i = 0; i < 9; i++) {
            if (board[i] == BoardSquare::X) state(i, 0) = 1.0f;
            else if (board[i] == BoardSquare::O) state(i, 0) = -1.0f;
            else state(i, 0) = 0.0f;
        }

        if (player == BoardSquare::O) state = -state;

        return state;
    }

    void print_board() {
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                std::cout << ' ' << square_to_char(at(x, y)) << ' ' << (x != 2 ? '|' : '\n');
            }

            if (y != 2) std::cout << "-----------" << std::endl;
        }
    }

    bool play_move(int x, int y) {
        if (at(x, y) != BoardSquare::EMPTY) return false;

        at(x, y) = next_player;
        next_player = next_player == BoardSquare::O ? BoardSquare::X : BoardSquare::O;

        return true;
    }
    bool play_move(int i) {
        if (at(i) != BoardSquare::EMPTY) return false;

        at(i) = next_player;
        next_player = next_player == BoardSquare::O ? BoardSquare::X : BoardSquare::O;

        return true;
    }

    // returns BoardSquare::EMPTY if there is no winner
    // returns the winner if there is a winner
    BoardSquare check_winner() {
        std::vector<std::vector<int>> lines {
            {0, 1, 2},
            {3, 4, 5},
            {6, 7, 8},
            {0, 3, 6},
            {1, 4, 7},
            {2, 5, 8},
            {0, 4, 8},
            {2, 4, 6}
        };

        for (auto line : lines) {
            if (at(line[0]) == BoardSquare::EMPTY) continue;
            if (at(line[0]) == at(line[1]) && at(line[1]) == at(line[2])) return at(line[0]);
        }

        return BoardSquare::EMPTY;
    }

    void restart() {
        for (int i = 0; i < 9; i++) board[i] = BoardSquare::EMPTY;
        next_player = BoardSquare::X;
    }
};