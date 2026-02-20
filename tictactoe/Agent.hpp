#pragma once
#include "./TicTacToe.hpp"
#include "./enums.hpp"

#include <concepts>
#include <array>
#include <random>
#include <exception>
#include <iostream>

template<typename T, int N, int W>
concept Agent = requires(T agent, TicTacToe<N, W>& game) {
    { agent.get_eval(game) } -> std::convertible_to<float>;
    { agent.get_move(game) } -> std::convertible_to<int>;
};

template<int N, int W, int B = 1048576>
struct MinimaxAgent {
    int depth;
    MinimaxAgent() : depth(9999) {}
    MinimaxAgent(int depth) : depth(depth) {}

    int get_eval(TicTacToe<N, W>& game) {
        return minimax(game, -9999, 9999, depth);
    }

    int get_move(TicTacToe<N, W>& game, int seed = std::random_device{}()) {
        bool maximising = game.next_player == BoardSquare::X;

        int max_val = -9999;
        int min_val = 9999;

        std::vector<int> max_moves;
        std::vector<int> min_moves;

        for (int move = 0; move < N * N; move++) {
            if (game.at(move) != BoardSquare::EMPTY) continue;
            
            game.play_move(move);
            float move_val = minimax(game, -9999, 9999, depth);
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
            throw std::invalid_argument("Board cannot be full");
        }

        std::mt19937 gen(seed);
        std::uniform_int_distribution<> distrib(0, moves.size() - 1);
        return moves[distrib(gen)];
    }

private:
    TranspositionTable<N, B> table;

    float block_heuristic(TicTacToe<N, W>& game, int move_x, int move_y, int move) {
        // check if it blocks a row (if everything other than us is the opponent)
        bool all_are_opponent = true;
        for (int i = 0; i < N; i++) {
            if (i == move_x) continue;
            if (game.at(i, move_y) == BoardSquare::EMPTY || game.at(i, move_y) == game.next_player) {all_are_opponent = false; break;}
        }
        if (all_are_opponent) return 1.0f;

        // check col
        all_are_opponent = true;
        for (int i = 0; i < N; i++) {
            if (i == move_y) continue;
            if (game.at(move_x, i) == BoardSquare::EMPTY || game.at(move_x, i) == game.next_player) {all_are_opponent = false; break;}
        }
        if (all_are_opponent) return 1.0f;

        // check top left to bottom right
        if (move_x == move_y) {
            all_are_opponent = true;
            for (int i = 0; i < N * N; i += N + 1) {
                if (i == move) continue;
                if (game.at(i) == BoardSquare::EMPTY || game.at(i) == game.next_player) {all_are_opponent = false; break;}
            }
            if (all_are_opponent) return 1.0f;
        }

        // check top right to bottom left
        if (move_x + move_y == N - 1) {
            all_are_opponent = true;
            for (int i = N - 1; i < N * N - 1; i += N - 1) {
                if (i == move) continue;
                if (game.at(i) == BoardSquare::EMPTY || game.at(i) == game.next_player) {all_are_opponent = false; break;}
            }
            if (all_are_opponent) return 1.0f;
        }

        return 0.0f;
    }

    float win_heuristic(TicTacToe<N, W>& game, int move) {
        game.play_move(move);
        BoardSquare winner = game.check_winner(move);
        game.unplay_move(move);

        if (winner == game.next_player) return 1.0f;
        if (winner == BoardSquare::EMPTY) return 0.0f;
        return -1.0f;
    }

    float fork_heuristic(TicTacToe<N, W>& game, int move_x, int move_y, int move) {
        int num_threats = 0;

        // check row
        int num_us = 1;
        for (int i = 0; i < N; i++) {
            if (i == move_x) continue;
            if (game.at(i, move_y) == BoardSquare::EMPTY) continue;
            if (game.at(i, move_y) == game.next_player) { num_us++; continue; }
            num_us = -1;
            break;
        }
        if (num_us == N - 1) num_threats++;

        // check col
        num_us = 1;
        for (int i = 0; i < N; i++) {
            if (i == move_y) continue;
            if (game.at(move_x, i) == BoardSquare::EMPTY) continue;
            if (game.at(move_x, i) == game.next_player) { num_us++; continue; }
            num_us = -1;
            break;
        }
        if (num_us == N - 1) num_threats++;

        // check top left to bottom right
        if (move_x == move_y) {
            num_us = 1;
            for (int i = 0; i < N * N; i += N + 1) {
                if (i == move) continue;
                if (game.at(i) == BoardSquare::EMPTY) continue;
                if (game.at(i) == game.next_player) { num_us++; continue; }
                num_us = -1;
                break;
            }
            if (num_us == N - 1) num_threats++;
        }

        // check top right to bottom left
        if (move_x + move_y == N - 1) {
            num_us = 1;
            for (int i = N - 1; i < N * N - 1; i += N - 1) {
                if (i == move) continue;
                if (game.at(i) == BoardSquare::EMPTY) continue;
                if (game.at(i) == game.next_player) { num_us++; continue; }
                num_us = -1;
                break;
            }
            if (num_us == N - 1) num_threats++;
        }

        if (num_threats > 1) return 1.0f;
        if (num_threats == 1) return 0.5f;
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
        // 2) create forks - 1.0f
        // 3) create threats - 0.5f
        // 3) play towards the middle - 0.2f

        float heuristic = 0.0f;

        int move_x = move % N;
        int move_y = move / N;

        heuristic = block_heuristic(game, move_x, move_y, move);
        if (heuristic == 1.0f) return heuristic;

        heuristic = win_heuristic(game, move);
        if (heuristic == 1.0f) return heuristic;
        if (heuristic == -1.0f) return heuristic;

        heuristic = fork_heuristic(game, move_x, move_y, move);
        if (heuristic == 1.0f) return heuristic;

        return heuristic;
    }

    float minimax(TicTacToe<N, W>& game, float alpha, float beta, int depth, int prev_move = -1) {
        if (prev_move != -1) {
            BoardSquare winner = game.check_winner(prev_move);
            if (winner != BoardSquare::EMPTY) return winner == BoardSquare::X ? 1 : -1;
        }

        if (depth <= 0) return 0;

        const float alpha_orig = alpha;
        const float beta_orig = beta;

        bool maximising = game.next_player == BoardSquare::X;
        
        std::array<std::pair<float, int>, N * N> move_buffer;
        int count = 0;

        int move_count = 0;

        for (int move = 0; move < N * N; move++) {
            if (game.at(move) != BoardSquare::EMPTY) continue;
            move_count++;
            move_buffer[count++] = std::pair<float, int>(get_heuristic(game, move), move);
        }

        if (move_count == 0) return 0;

        std::sort(move_buffer.begin(), move_buffer.begin() + count, [](std::pair<float, int> a, std::pair<float, int> b) { return a.first > b.first; });

        float min_val = 9999;
        float max_val = -9999;

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

        for (int i = 0; i < count; i++) {
            int move = move_buffer[i].second;

            game.play_move(move);
            
            float val = minimax(game, alpha, beta, depth - 1, move);

            game.unplay_move(move);
            
            if (maximising) {
                max_val = std::max(val, max_val);
                alpha = std::max(val, alpha);
            } else {
                min_val = std::min(val, min_val);
                beta = std::min(val, beta);
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

template<int N, int W, int S = 0>
struct RandomAgent {
    float get_eval(TicTacToe<N, W>& game) {
        return 0.0;
    }

    int get_move(TicTacToe<N, W>& game) {
        std::vector<int> moves;
        for (int i = 0; i < N * N; i++) {
            if (game.at(i) != BoardSquare::EMPTY) continue;
            moves.push_back(i);
        }

        std::mt19937 mt(S);
        std::uniform_int_distribution<> distrib(0, moves.size() - 1);
        return moves[distrib(mt)];
    }
};

template<int N, int W, int S = 0>
struct HumanAgent {
    float get_eval(TicTacToe<N, W>& game) {

        float user_eval;
        do {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "If -1 is winning for O and 1 is winning for X, how would evaluate the game?" << std::endl;
        } while (!(std::cin >> user_eval));

        return user_eval;
    }

    int get_move(TicTacToe<N, W>& game) {
        game.print_board();

        int move_x;
        int move_y;

        while (true) {
            std::cout << "Move x: ";
            while (!(std::cin >> move_x)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid number. Move x: ";
            }

            std::cout << "Move y: ";
            while (!(std::cin >> move_y)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid number. Move y: ";
            }

            if (move_x < 0 || move_x >= N || move_y < 0 || move_y >= N) std::cout << "Invalid move" << std::endl;
            else if (game.at(move_x, move_y) != BoardSquare::EMPTY)  std::cout << "Invalid move" << std::endl;
            else break;
        }

        return move_y * N + move_x;
    }
};