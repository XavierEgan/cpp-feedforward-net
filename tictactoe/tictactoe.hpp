#pragma once
#include "../Eigen/Dense"
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <algorithm>

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

template<int N>
struct TicTacToe {
    BoardSquare next_player;
    int num_pieces = N;
    std::array<BoardSquare, N * N> board;

    TicTacToe() {
        board.fill(BoardSquare::EMPTY);
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
    BoardSquare check_winner() const {
        // check rows
        for (int y = 0; y < N; y++) {
            BoardSquare first_square = at(0, y);
            if (first_square == BoardSquare::EMPTY) continue;

            bool row_win = true;
            for (int x = 1; x < N; x++) {
                if (at(x, y) != first_square) {
                    row_win = false;
                    break;
                }
            }

            if (row_win) return first_square;
        }

        // check columns
        for (int x = 0; x < N; x++) {
            BoardSquare first_square = at(x, 0);
            if (first_square == BoardSquare::EMPTY) continue;

            bool col_win = true;
            for (int y = 1; y < N; y++) {
                if (at(x, y) != first_square) {
                    col_win = false;
                    break;
                }
            }

            if (col_win) return first_square;
        }

        // check diagonal top-left to bottom-right
        BoardSquare first_square = at(0, 0);
        if (first_square != BoardSquare::EMPTY) {
            bool diag_win = true;
            for (int i = 1; i < N; i++) {
                if (at(i, i) != first_square) {
                    diag_win = false;
                    break;
                }
            }

            if (diag_win) return first_square;
        }

        // check diagonal top-right to bottom-left
        first_square = at(N - 1, 0);
        if (first_square != BoardSquare::EMPTY) {
            bool diag_win = true;
            for (int i = 1; i < N; i++) {
                if (at(N - 1 - i, i) != first_square) {
                    diag_win = false;
                    break;
                }
            }

            if (diag_win) return first_square;
        }

        return BoardSquare::EMPTY;
    }

    // returns BoardSquare::EMPTY if there is no winner
    // returns the winner if there is a winner
    // most_recent_move makes it more efficient
    BoardSquare check_winner(int most_recent_move) const {
        int move_x = most_recent_move % N;
        int move_y = most_recent_move / N;

        BoardSquare start_square = at(most_recent_move);
        if (start_square == BoardSquare::EMPTY) return BoardSquare::EMPTY;

        // check col
        bool col_win = true;
        for (int i = 0; i < N; i++) {
            if (at(move_x, i) != start_square) col_win = false;
        }
        if (col_win) return start_square;

        // check row
        bool row_win = true;
        for (int i = 0; i < N; i++) {
            if (at(i, move_y) != start_square) row_win = false;
        }
        if (row_win) return start_square;

        // check top left to bottom right
        if (move_x == move_y) {
            bool diag_tlbr_win = true;
            for (int i = 0; i < N * N; i += N + 1) {
                if (at(i) != start_square) diag_tlbr_win = false;
            }
            if (diag_tlbr_win) return start_square;
        }

        // check top right to bottom left
        if (move_x + move_y == N - 1) {
            bool diag_trbl_win = true;
            for (int i = N - 1; i < N * N - 1; i += N - 1) {
                if (at(i) != start_square) diag_trbl_win = false;
            }
            if (diag_trbl_win) return start_square;
        }

        return BoardSquare::EMPTY;
    }
    
    void restart() {
        for (int i = 0; i < N * N; i++) board[i] = BoardSquare::EMPTY;
        next_player = BoardSquare::X;
    }

    int get_board_eval(int max_depth = 9999) {
        bool maximising = next_player == BoardSquare::X;
        return minimax(maximising, -9999, 9999, max_depth - 1);
    }

    int get_move_eval(int move, int max_depth = 9999) {
        bool maximising = next_player == BoardSquare::X;
        play_move(move);
        int move_val = minimax(!maximising, -9999, 9999, max_depth);
        unplay_move(move);
        return move_val;
    }

    int get_best_move(int max_depth = 9999, int seed = std::random_device{}()) {
        bool maximising = next_player == BoardSquare::X;

        if (maximising) {
            int max_val = -9999;
            int max_move = -1;
            std::vector<int> max_moves;

            for (int move = 0; move < N * N; move++) {
                if (board[move] != BoardSquare::EMPTY) continue;
                
                int move_val = get_move_eval(move, max_depth - 1);

                std::cout << "Move: " << move << " Val: " << move_val << std::endl;

                if (move_val > max_val) {
                    max_val = move_val;
                    max_move = move;
                    max_moves.clear();
                }
                if (move_val == max_val) max_moves.push_back(move);
            }

            std::mt19937 gen(seed);
            std::uniform_int_distribution<> distrib(0, max_moves.size() - 1);
            return max_moves[distrib(gen)];
            
        } else {
            int min_val = 9999; 
            int min_move = -1;
            std::vector<int> min_moves;

            for (int move = 0; move < N * N; move++) {
                if (board[move] != BoardSquare::EMPTY) continue;
                
                int move_val = get_move_eval(move, max_depth - 1);

                std::cout << "Move: " << move << " Val: " << move_val << std::endl;

                if (move_val < min_val) {
                    min_val = move_val;
                    min_move = move;
                    min_moves.clear();
                }
                if (move_val == min_val) min_moves.push_back(move);
            }

            std::mt19937 gen(seed);
            std::uniform_int_distribution<> distrib(0, min_moves.size() - 1);
            return min_moves[distrib(gen)];
        }
    }

private:
    int minimax(bool maximising, int alpha, int beta, int depth, int prev_move = -1) {
        if (prev_move == -1) {
            BoardSquare winner = check_winner();
            if (winner != BoardSquare::EMPTY) return winner == BoardSquare::X ? 1 : -1;
        } else {
            BoardSquare winner = check_winner(prev_move);
            if (winner != BoardSquare::EMPTY) return winner == BoardSquare::X ? 1 : -1;
        }

        if (depth <= 0) return 0;

        if (maximising) {
            int max_val = -9999;
            int move_count = 0;
            for (int move = 0; move < N * N; move++) {
                if (board[move] != BoardSquare::EMPTY) continue;

                move_count++;

                play_move(move);
                int val = minimax(!maximising, alpha, beta, depth - 1, move);
                unplay_move(move);
                
                max_val = std::max(val, max_val);
                alpha = std::max(val, alpha);

                if (beta <= alpha) break;
            }
            if (move_count == 0) return 0;
            return max_val;

        } else {
            int min_val = 9999;
            int move_count = 0;
            for (int move = 0; move < N * N; move++) {
                if (board[move] != BoardSquare::EMPTY) continue;

                move_count++;

                play_move(move);
                int val = minimax(!maximising, alpha, beta, depth - 1, move);
                unplay_move(move);
                
                min_val = std::min(val, min_val);
                beta = std::min(val, beta);

                if (beta <= alpha) break;
            }
            if (move_count == 0) return 0;
            return min_val;
        }
    } 
};