#pragma once
#include "../Eigen/Dense"
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>

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

    void play_move(Eigen::MatrixXf move_probabilities, float epsilon=0.0f) {
        int move_index;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> uniform(0.0, 1.0);

        if (uniform(gen) < epsilon) {
            // get all valid moves
            std::vector<int> valid_moves;
            for (int i = 0; i < N * N; i++) {
                if (at(i) == BoardSquare::EMPTY) valid_moves.push_back(i);
            }
            
            // pick a random valid move
            std::uniform_int_distribution<> distrib(0, valid_moves.size() - 1);
            move_index = valid_moves[distrib(gen)];
        } else {
            std::discrete_distribution<> distrib(move_probabilities.data(), move_probabilities.data() + move_probabilities.size());
            move_index = distrib(gen);
        }
        
        // try do the move
        bool move_succeeded = play_move(move_index);
        
        if (!move_succeeded) {
            // that move was invalid, sort the matrix and play moves in order of likelyhood untill one works
            std::vector<std::pair<float, int>> probabilities(N * N);
            for (int i = 0; i < N * N; i++) {
                probabilities[i] = std::pair<float, int>(move_probabilities(i), i);
            }

            std::sort(probabilities.begin(), probabilities.end(), [](const auto& a, const auto&  b){ return a.first > b.first; });

            for (int i = 1; i < N * N; i++) {
                if (play_move(probabilities[i].second)) {
                    move_index = probabilities[i].second;
                    break;
                }
            }
        }
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

    void restart() {
        for (int i = 0; i < N * N; i++) board[i] = BoardSquare::EMPTY;
        next_player = BoardSquare::X;
    }

    int minimax(bool maximising) {
        BoardSquare winner = check_winner();
        if (winner != BoardSquare::EMPTY) return winner == BoardSquare::X ? 1 : -1;

        int max_val = -9999;
        int min_val = 9999;

        int move_count = 0; // check for draw

        for (int move = 0; move < N * N; move++) {
            if (board[move] != BoardSquare::EMPTY) continue;
            move_count++;

            play_move(move);
            int val = minimax(!maximising);
            unplay_move(move);
            
            if (val > max_val) max_val = val;
            if (val < min_val) min_val = val;
        }

        if (move_count == 0) return 0;
        return maximising ? max_val : min_val;
    } 

    int get_best_move() {
        bool maximising = next_player == BoardSquare::X;

        int max_val = -9999;
        int min_val = 9999; 

        int max_move = -1;
        int min_move = -1;

        for (int move = 0; move < N * N; move++) {
            if (board[move] != BoardSquare::EMPTY) continue;
            
            play_move(move);
            int move_val = minimax(!maximising);
            unplay_move(move);

            std::cout << "Move: " << move << " Val: " << move_val << std::endl;

            if (move_val > max_val) {
                max_val = move_val;
                max_move = move;
            }

            if (move_val < min_val) {
                min_val = move_val;
                min_move = move;
            }
        }

        return maximising ? max_move : min_move;
    }
};