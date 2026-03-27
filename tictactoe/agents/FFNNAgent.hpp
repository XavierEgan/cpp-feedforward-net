#pragma once
#include "../TicTacToe.hpp"
#include "../enums.hpp"
#include "../TranspositionTable.hpp"
#include "../../FFNN.hpp"

#include <concepts>
#include <array>
#include <random>
#include <exception>
#include <iostream>
#include <string>
#include <chrono>


template<int N, int W, int B = 1048576 * 16>
struct FFNNAgent {
    int depth;
    std::string name;
    FFNN eval_ffnn;

    FFNNAgent(FFNN eval_ffnn) : depth(9999), name("Unnamed FFNNAgent"), eval_ffnn(eval_ffnn) {}
    FFNNAgent(int depth, FFNN eval_ffnn) : depth(depth), name("Unnamed FFNNAgent"), eval_ffnn(eval_ffnn) {}
    FFNNAgent(int depth, std::string name, FFNN eval_ffnn) : depth(depth), name(name), eval_ffnn(eval_ffnn) {}

    float get_eval(TicTacToe<N, W>& game) {
        return minimax(game, -9999, 9999, depth);
    }

    int get_move(TicTacToe<N, W>& game, int seed = std::random_device{}()) {
        bool maximising = game.next_player == BoardSquare::X;

        float max_val = -9999.0f;
        float min_val = 9999.0f;

        std::vector<int> max_moves;
        std::vector<int> min_moves;

        for (int move = 0; move < N * N; move++) {
            if (game.at(move) != BoardSquare::EMPTY) continue;
            
            game.play_move(move);
            float move_val = minimax(game, -9999, 9999, depth, move);
            game.unplay_move(move);

            if (maximising && move_val > max_val) {
                max_val = move_val;
                max_moves.clear();
            }
            if (!maximising && move_val < min_val) {
                min_val = move_val;
                min_moves.clear();
            }

            if (maximising && move_val == max_val) max_moves.push_back(move);
            if (!maximising && move_val == min_val) min_moves.push_back(move);
        }

        std::vector<int>& moves = maximising ? max_moves : min_moves;

        if (moves.size() == 0) {
            game.print_board();
            throw std::invalid_argument("Board cannot be full");
        }

        std::mt19937 gen(seed);
        std::uniform_int_distribution<> distrib(0, moves.size() - 1);

        return moves[distrib(gen)];
    }

    std::string& get_name() {
        return name;
    }
private:
    TranspositionTable<N, B> table;

    float get_heuristic(TicTacToe<N, W>& game, int move) {
        // heuristic_ffnn.forward(game.get_board_state());
        return 0.0f;
    }

    float get_static_eval(TicTacToe<N, W>& game, int prev_move = -1) {
        return eval_ffnn.forward(game.get_board_state())(0, 0);
    }

    float minimax(TicTacToe<N, W>& game, float alpha, float beta, int depth, int prev_move = -1) {
        if (depth <= 0) return get_static_eval(game, prev_move);

        if (prev_move != -1) {
            BoardSquare winner = game.check_winner(prev_move);
            if (winner == BoardSquare::X) return 1.0f;
            if (winner == BoardSquare::O) return -1.0f;
        }

        long long key = table.hash(game.board, game.next_player);
        if (table.contains(key)) {
            TranspositionTableEntry entry = table.get(key);
            if (entry.depth >= depth) {
                if (entry.type == NodeType::EXACT) return entry.score;
                if (entry.type == NodeType::LOWER) alpha = std::max(alpha, entry.score);
                else beta = std::min(beta, entry.score);
                
                if (alpha >= beta) return entry.score;
            }
        }

        const float alpha_orig = alpha;
        const float beta_orig = beta;

        bool maximising = game.next_player == BoardSquare::X;
        
        std::array<std::pair<float, int>, N * N> move_buffer;
        int count = 0;

        for (int move = 0; move < N * N; move++) {
            if (game.at(move) != BoardSquare::EMPTY) continue;
            move_buffer[count++] = std::pair<float, int>(get_heuristic(game, move), move);
        }

        if (count == 0) return 0;

        // TODO: uncomment this if you use heuristic
        // std::sort(move_buffer.begin(), move_buffer.begin() + count, [](std::pair<float, int> a, std::pair<float, int> b) { return a.first > b.first; });

        float min_val = 9999;
        float max_val = -9999;

        for (int i = 0; i < count; i++) {
            int move = move_buffer[i].second;

            game.play_move(move);
            
            float val = minimax(game, alpha, beta, depth - 1, move);

            game.unplay_move(move);
            
            if (maximising) {
                max_val = std::max(val, max_val);
                alpha = std::max(alpha, val);
            } else {
                min_val = std::min(val, min_val);
                beta = std::min(beta, val);
            }

            if (beta <= alpha) break;
        }

        float best = maximising ? max_val : min_val;

        NodeType type = NodeType::EXACT;
        if (best <= alpha_orig) type = NodeType::UPPER;
        else if (best >= beta_orig) type = NodeType::LOWER;
        
        table.insert(key, TranspositionTableEntry{best, depth, type});

        return best;
    }
};