#pragma once
#include "../tictactoe.hpp"
#include "../enums.hpp"
#include "../TranspositionTable.hpp"
#include "../../FFNN.hpp"
#include "../../Workspace.hpp"

#include <concepts>
#include <array>
#include <random>
#include <exception>
#include <iostream>
#include <string>
#include <chrono>

template<int N, int W, int B = 1048576 * 16>
struct FFNNAgent {
    std::chrono::duration<double, std::milli> max_move_time;
    std::string name;
    FFNN ffnn;
    ForwardWorkspace ws;   // reused across evals, allocation-free after the first call

    FFNNAgent(FFNN ffnn) : max_move_time(1), name("Unnamed FFNNAgent"), ffnn(ffnn), ws(ForwardWorkspace::from_shape(ffnn.network_shape)) {}
    FFNNAgent(double max_move_time, FFNN ffnn) : max_move_time(max_move_time), name("Unnamed FFNNAgent"), ffnn(ffnn), ws(ForwardWorkspace::from_shape(ffnn.network_shape)) {}
    FFNNAgent(double max_move_time, std::string name, FFNN ffnn)  : max_move_time(max_move_time), name(name), ffnn(ffnn), ws(ForwardWorkspace::from_shape(ffnn.network_shape)) {}

    float get_eval(TicTacToe<N, W>& game) {
        // TODO: fix ts
        table.map.clear();
        return minimax(game, -9999, 9999, 1);
    }

    std::string& get_name() {
        return name;
    }

    int last_depth = -1;

    int get_last_depth() const { return last_depth; }

    int get_move(TicTacToe<N, W>& game, int seed = std::random_device{}()) {
        // try get move at depth 1, then depth 2 etc and use the last move that didnt time out
        std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(max_move_time);

        int prev_move = -1;
        int cur_move;
        
        for (int depth = 0; true; depth++) {
            cur_move = get_move_at_depth(game, depth, deadline, seed);

            if (std::chrono::steady_clock::now() >= deadline) {
                if (prev_move == -1) {
                    std::cout << "Agent reached deadline before finishing depth zero" << std::endl;

                    // get the first move we can
                    for (int move = 0; move < N * N; move++) {
                        if (game.at(move) != BoardSquare::EMPTY) continue;
                        return move;
                    }
                }
                return prev_move;
            }

            prev_move = cur_move;
            last_depth = depth;
            if (depth >= N * N) break;
        }
        return prev_move;
    }
private:
    TranspositionTable<N, B> table;

    int get_move_at_depth(TicTacToe<N, W>& game, int depth, std::chrono::steady_clock::time_point deadline, int seed = std::random_device{}()) {
        table.map.clear();
        bool maximising = game.next_player == BoardSquare::X;

        float max_val = -9999.0f;
        float min_val = 9999.0f;

        std::vector<int> max_moves;
        std::vector<int> min_moves;

        for (int move = 0; move < N * N; move++) {
            if (std::chrono::steady_clock::now() >= deadline)
                return 0;

            if (game.at(move) != BoardSquare::EMPTY) continue;
            
            game.play_move(move);
            float move_val = minimax(game, -9999, 9999, depth, deadline, move);
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

    float block_heuristic(TicTacToe<N, W>& game, int move) {
        // returns 1.0 if playing move blocks an opponent W-1-in-a-row threat
        for (int wi : game.cell_to_windows[move]) {
            const auto& ws = game.window_states[wi];
            if (game.next_player == BoardSquare::X && ws.num_o == W - 1 && ws.num_x == 0) return 1.0f;
            if (game.next_player == BoardSquare::O && ws.num_x == W - 1 && ws.num_o == 0) return 1.0f;
        }
        return 0.0f;
    }

    float win_heuristic(TicTacToe<N, W>& game, int move) {
        // Returns 1.0 if playing move immediately wins
        for (int wi : game.cell_to_windows[move]) {
            const auto& ws = game.window_states[wi];
            if (game.next_player == BoardSquare::X && ws.num_x == W - 1 && ws.num_o == 0) return 1.0f;
            if (game.next_player == BoardSquare::O && ws.num_o == W - 1 && ws.num_x == 0) return 1.0f;
        }
        return 0.0f;
    }

    float fork_heuristic(TicTacToe<N, W>& game, int move) {
        // Returns 0.9 if playing move creates >=2 new threats , 0.5 for one new threat
        if constexpr (W < 3) return 0.0f;
        int new_threats = 0;
        for (int wi : game.cell_to_windows[move]) {
            const auto& ws = game.window_states[wi];
            if (game.next_player == BoardSquare::X && ws.num_o == 0 && ws.num_x == W - 2) new_threats++;
            if (game.next_player == BoardSquare::O && ws.num_x == 0 && ws.num_o == W - 2) new_threats++;
        }
        if (new_threats >= 2) return 0.9f;
        if (new_threats == 1) return 0.5f;
        return 0.0f;
    }

    float center_heuristic(int move_x, int move_y) {
        float cx = (N - 1) * 0.5f;
        float cy = (N - 1) * 0.5f;

        float dist = std::abs(move_x - cx) + std::abs(move_y - cy);
        float max_dist = cx + cy;

        if (max_dist <= 0.0f) return 1.0f;
        return 1.0f - (dist / max_dist);
    }

    // returns how good the move is under some heuristics (larger is better)
    // max return is 1
    float get_heuristic(TicTacToe<N, W>& game, int move) {
        // heuristics:
        // 1) block N in a rows - 1.0f
        // 2) create forks - 0.9f
        // 3) create threats - 0.5f
        // 4) play towards the middle - 0.2f (defined but unused)
        float heuristic = 0.0f;
        heuristic += block_heuristic(game, move);
        heuristic += win_heuristic(game, move);
        heuristic += fork_heuristic(game, move);
        return heuristic;
    }

    float get_static_eval(TicTacToe<N, W>& game, int prev_move = -1) {
        // ffnn eval
        return ffnn.forward(game.get_board_state(), ws)(0, 0);
    }

    float minimax(TicTacToe<N, W>& game, float alpha, float beta, int depth, std::chrono::steady_clock::time_point deadline, int prev_move = -1) {
        // since were not using it anyway if we hit deadline
        if (std::chrono::steady_clock::now() >= deadline) return 0;
        if (depth <= 0) return get_static_eval(game, prev_move);

        if (prev_move != -1) {
            BoardSquare winner = game.check_winner(prev_move);

            constexpr float eps = 0.001f;
            if (winner == BoardSquare::X) return  1.0f + depth * eps;
            if (winner == BoardSquare::O) return -1.0f - depth * eps;
        }

        long long key = game.hash_val;
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

        std::sort(move_buffer.begin(), move_buffer.begin() + count, [](std::pair<float, int> a, std::pair<float, int> b) { return a.first > b.first; });

        float min_val = 9999;
        float max_val = -9999;

        for (int i = 0; i < count; i++) {
            if (std::chrono::steady_clock::now() >= deadline)
                return 0;

            int move = move_buffer[i].second;

            game.play_move(move);
            
            float val = minimax(game, alpha, beta, depth - 1, deadline, move);

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