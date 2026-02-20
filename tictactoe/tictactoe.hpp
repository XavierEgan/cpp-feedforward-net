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
template<int N>
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
        winner = BoardSquare::EMPTY;
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

    Eigen::MatrixXf get_board_state(BoardSquare player) {
        Eigen::MatrixXf state(N * N, 1);
        for (int i = 0; i < N * N; i++) {
            if (board[i] == BoardSquare::X) state(i, 0) = 1.0f;
            else if (board[i] == BoardSquare::O) state(i, 0) = -1.0f;
            else state(i, 0) = 0.0f;
        }

        if (player == BoardSquare::O) state = -state;

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
    BoardSquare get_winner() const {
        return winner;
    }

    // returns BoardSquare::EMPTY if there is no winner
    // returns the winner if there is a winner
    // most_recent_move makes it more efficient
    BoardSquare check_winner(int most_recent_move) {
        if (get_winner() != BoardSquare::EMPTY) {
            throw std::runtime_error("Winner should be empty when checking winner");
        }

        int move_x = most_recent_move % N;
        int move_y = most_recent_move / N;

        BoardSquare start_square = at(most_recent_move);
        if (start_square == BoardSquare::EMPTY) return BoardSquare::EMPTY;

        // check col
        bool col_win = true;
        for (int i = 0; i < N; i++) {
            if (at(move_x, i) != start_square) col_win = false;
        }
        if (col_win) { return start_square; winner = start_square; }

        // check row
        bool row_win = true;
        for (int i = 0; i < N; i++) {
            if (at(i, move_y) != start_square) row_win = false;
        }
        if (row_win) { return start_square; winner = start_square; }

        // check top left to bottom right
        if (move_x == move_y) {
            bool diag_tlbr_win = true;
            for (int i = 0; i < N * N; i += N + 1) {
                if (at(i) != start_square) {diag_tlbr_win = false; break;}
            }
            if (diag_tlbr_win) { return start_square; winner = start_square; }
        }

        // check top right to bottom left
        if (move_x + move_y == N - 1) {
            bool diag_trbl_win = true;
            for (int i = N - 1; i < N * N - 1; i += N - 1) {
                if (at(i) != start_square) {diag_trbl_win = false; break;}
            }
            if (diag_trbl_win) { return start_square; winner = start_square; }
        }

        return BoardSquare::EMPTY;
    }
    

private:
    BoardSquare winner = BoardSquare::EMPTY;
};