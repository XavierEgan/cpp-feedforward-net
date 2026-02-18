#pragma once
#include "../Eigen/Dense"
#include <iostream>
#include <string>
#include <vector>
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
    std::vector<BoardSquare> board;

    // board state before the player takes a move
    std::vector<Eigen::MatrixXf> player_x_game_history;
    std::vector<Eigen::MatrixXf> player_o_game_history;
    std::vector<int> player_x_moves;
    std::vector<int> player_o_moves;

    TicTacToe() {
        board = std::vector<BoardSquare>(N * N, BoardSquare::EMPTY);
        next_player = BoardSquare::X;
    }

    BoardSquare at(int index) const {
        return board.at(index);
    }
    BoardSquare& at(int index) {
        return board.at(index);
    }
    BoardSquare at(int x, int y) const {
        return board.at(y * N + x);
    }
    BoardSquare& at(int x, int y) {
        return board.at(y * N + x);
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

        // note down the board state and move played BEFORE updating board (because we need to train on the board state before the move to predict the move)
        if (next_player == BoardSquare::X) {
            player_x_game_history.push_back(get_board_state(BoardSquare::X));
            player_x_moves.push_back(i);
        } else {
            player_o_game_history.push_back(get_board_state(BoardSquare::O));
            player_o_moves.push_back(i);
        }

        at(i) = next_player;

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
    BoardSquare check_winner() {
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
    
    void append_player_positive_example(std::vector<Eigen::MatrixXf>& inputs, std::vector<Eigen::MatrixXf>& targets, BoardSquare player) {
        std::vector<Eigen::MatrixXf>& player_game_history = player == BoardSquare::X ? player_x_game_history : player_o_game_history;
        std::vector<int>& player_moves = player == BoardSquare::X ? player_x_moves : player_o_moves;

        for (size_t i = 0; i < player_game_history.size(); i++) {
            // add the board state as an input
            inputs.push_back(player_game_history[i]);

            // add the move played as a target
            Eigen::MatrixXf target = Eigen::MatrixXf::Zero(N * N, 1);
            target(player_moves[i], 0) = 1.0f;
            targets.push_back(target);
        }
    }
    void append_player_x_positive_example(std::vector<Eigen::MatrixXf>& inputs, std::vector<Eigen::MatrixXf>& targets) {
        append_player_positive_example(inputs, targets, BoardSquare::X);
    }
    void append_player_o_positive_example(std::vector<Eigen::MatrixXf>& inputs, std::vector<Eigen::MatrixXf>& targets) {        
        append_player_positive_example(inputs, targets, BoardSquare::O);
    }

    void restart() {
        for (int i = 0; i < N * N; i++) board[i] = BoardSquare::EMPTY;
        next_player = BoardSquare::X;

        player_x_game_history.clear();
        player_o_game_history.clear();
        player_x_moves.clear();
        player_o_moves.clear();
    }
};