#pragma once
#include "./enums.hpp"
#include "./TranspositionTable.hpp"

#include "../Eigen/Dense"
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <unordered_map>

// NxN board, W win length
template<int N, int W>
struct TicTacToe {
    BoardSquare next_player;
    int num_pieces = N;
    std::array<BoardSquare, N * N> board;

    TicTacToe() {
        board.fill(BoardSquare::EMPTY);
        next_player = BoardSquare::X;
        if (N < 2) {
            throw std::length_error("N should not be less than 2");
        }
    }

    void restart() {
        for (int i = 0; i < N * N; i++) board[i] = BoardSquare::EMPTY;
        next_player = BoardSquare::X;
    }

    BoardSquare at(int index) const {
        return board[index];
    }
    BoardSquare& at(int index) {
        return board[index];
    }
    BoardSquare at(int x, int y) const {
        return board[y * N + x];
    }
    BoardSquare& at(int x, int y) {
        return board[y * N + x];
    }

    /*
    return a matrix representation of the board with
    1 = X
    -1 = O
    0 = blank
    */
    Eigen::MatrixXf get_board_state() {
        Eigen::MatrixXf state(N * N, 1);
        for (int i = 0; i < N * N; i++) {
            if (board[i] == BoardSquare::X) state(i, 0) = 1.0f;
            else if (board[i] == BoardSquare::O) state(i, 0) = -1.0f;
            else state(i, 0) = 0.0f;
        }

        return state;
    }

    void print_board() {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                std::cout << ' ' << square_to_char(at(x, y)) << ' ' << (x != N - 1 ? '|' : '\n');
            }

            if (y != N - 1) std::cout << std::string(N * 4 - 1, '-') << std::endl;
        }
    }

    bool play_move(int i) {
        if (at(i) != BoardSquare::EMPTY) return false;

        at(i) = next_player;
        next_player = next_player == BoardSquare::O ? BoardSquare::X : BoardSquare::O;

        return true;
    }

    bool unplay_move(int i) {
        if (at(i) == BoardSquare::EMPTY) return false;

        at(i) = BoardSquare::EMPTY;
        next_player = next_player == BoardSquare::O ? BoardSquare::X : BoardSquare::O;

        return true;
    }

    // returns BoardSquare::EMPTY if there is no winner
    // returns the winner if there is a winner
    BoardSquare check_winner(int most_recent_move) {
        int x = most_recent_move % N;
        int y = most_recent_move / N;

        BoardSquare p = at(most_recent_move);
        if (p == BoardSquare::EMPTY) return BoardSquare::EMPTY;

        auto count_dir = [&](int dx, int dy) {
            int c = 0;
            int cx = x + dx, cy = y + dy;
            while (cx >= 0 && cx < N && cy >= 0 && cy < N && at(cx, cy) == p) {
                ++c;
                cx += dx;
                cy += dy;
            }
            return c;
        };

        // horizontal
        if (1 + count_dir(-1, 0) + count_dir(1, 0) >= W) return p;
        // vertical
        if (1 + count_dir(0, -1) + count_dir(0, 1) >= W) return p;
        // diagonal TL-BR
        if (1 + count_dir(-1, -1) + count_dir(1, 1) >= W) return p;
        // diagonal TR-BL
        if (1 + count_dir(1, -1) + count_dir(-1, 1) >= W) return p;

        return BoardSquare::EMPTY;
    }
};