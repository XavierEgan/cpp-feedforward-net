#pragma once
#include "./TicTacToe.hpp"
#include "./enums.hpp"

#include <concepts>
#include <array>
#include <random>
#include <exception>
#include <iostream>
#include <string>
#include <chrono>

template<typename T, int N, int W>
concept Agent = requires(T agent, TicTacToe<N, W>& game) {
    { agent.get_eval(game) } -> std::convertible_to<float>;
    { agent.get_move(game) } -> std::convertible_to<int>;
    { agent.get_name() } -> std::convertible_to<std::string>;
};

template<int N, int W, int B = 1048576>
struct MinimaxAgent {
    int depth;
    std::string name;

    MinimaxAgent() : depth(9999), name("Unnamed MinimaxAgent") {}
    MinimaxAgent(int depth) : depth(depth), name("Unnamed MinimaxAgent") {}
    MinimaxAgent(int depth, std::string name) : depth(depth), name(name) {}

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
        auto check_direction = [&](int dx, int dy) -> int {
            // keep count of how many we actually have in a row
            // note how many we could get in a row in +dx +dy with one added
            // note how many we could get in a row in -dy -dy with one added

            int x = move_x + dx;
            int y = move_y + dy;

            int num_in_a_row = 1; // starts at 1 because of the move were checking
            int possible_pos_in_a_row = 0;
            int possible_neg_in_a_row = 0;

            while (x >= 0 && x < N && y >= 0 && y < N) {
                if (possible_pos_in_a_row == 0) {
                    if (game.at(x, y) == game.next_player) num_in_a_row++;
                    else if (game.at(x, y) == BoardSquare::EMPTY) possible_pos_in_a_row++;
                } else {
                    if (game.at(x, y) == game.next_player) possible_pos_in_a_row++;
                    else if (game.at(x, y) == BoardSquare::EMPTY) possible_pos_in_a_row++;
                }

                x += dx;
                y += dy;
            }

            x = move_x - dx;
            y = move_y - dy;

            while (x >= 0 && x < N && y >= 0 && y < N) {
                if (possible_neg_in_a_row == 0) {
                    if (game.at(x, y) == game.next_player) num_in_a_row++;
                    else if (game.at(x, y) == BoardSquare::EMPTY) possible_neg_in_a_row++;
                    else break;
                } else {
                    if (game.at(x, y) == game.next_player) possible_neg_in_a_row++;
                    else if (game.at(x, y) == BoardSquare::EMPTY) possible_neg_in_a_row++;
                    else break;
                }

                x -= dx;
                y -= dy;
            }

            if (num_in_a_row + possible_pos_in_a_row >= W && num_in_a_row + possible_neg_in_a_row >= W) return 2;
            else if (num_in_a_row + possible_pos_in_a_row >= W) return 1;
            else if (num_in_a_row + possible_neg_in_a_row >= W) return 1;
            else return 0;
        };

        int fork_count = 0;
        fork_count += check_direction(1, 0);
        fork_count += check_direction(0, 1);
        fork_count += check_direction(1, 1);
        fork_count += check_direction(-1, 1);

        if (fork_count >= 2) return 0.9f;
        if (fork_count == 1) return 0.5f;
        else return 0.0f;
    }

    float center_heuristic(int move_x, int move_y) {
        float cx = (N - 1) * 0.5f;
        float cy = (N - 1) * 0.5f;

        float dist = std::abs(move_x - cx) + std::abs(move_y - cy);
        float max_dist = cx + cy;

        if (max_dist <= 0.0f) return 1.0f;
        return 1.0f - (dist / max_dist);
    }

    // returns how good the move is under some heuristics(larger is better)
    // max return is 1
    float get_heuristic(TicTacToe<N, W>& game, int move) {
        // heuristics:
        // 1) block N in a rows - 1.0f
        // 2) create forks - 0.9f
        // 3) create threats - 0.5f
        // 3) play towards the middle - 0.2f

        float heuristic = 0.0f;

        int move_x = move % N;
        int move_y = move / N;

        heuristic += block_heuristic(game, move_x, move_y, move);

        heuristic += win_heuristic(game, move);

        heuristic += fork_heuristic(game, move_x, move_y, move);

        return heuristic;
    }

    float get_static_eval(TicTacToe<N, W>& game, int prev_move = -1) {
        // static eval of the board, max = +-1.0
        // x = +ve
        // o = -ve
        constexpr float o_const = -1.0f;
        constexpr float x_const = 1.0f;
        constexpr float win = 1.0f;
        constexpr float threat = 0.1f; // if there are 10 threats then lets be honest its probably a win for you
        constexpr float central_piece = 0.01f; // a single threat is vastly more valuable than a central piece

        float eval = 0.0f;

        if (prev_move != -1) {
            BoardSquare winner = game.check_winner(prev_move);
            if (winner == BoardSquare::O) return o_const * win;
            else if (winner == BoardSquare::X) return win;
        }

        auto eval_window = [&](int x, int y, int dx, int dy) -> float {
            int num_x = 0;
            int num_o = 0;
            int num_empty = 0;

            for (int i = 0; i < W; i++) {
                BoardSquare sq = game.at(x + i * dx, y + i * dy);
                if (sq == BoardSquare::X) num_x++;
                else if (sq == BoardSquare::O) num_o++;
                else num_empty++;
            }

            if (num_x > 0 && num_o > 0) return 0.0f;
            if (num_x == 0 && num_o == 0) return 0.0f;

            if (num_o == 0) {
                if (num_x == W - 1 && num_empty == 1) return threat;
                return threat * (static_cast<float>(num_x) / static_cast<float>(W));
            }

            if (num_x == 0) {
                if (num_o == W - 1 && num_empty == 1) return -threat;
                return -threat * (static_cast<float>(num_o) / static_cast<float>(W));
            }

            return 0.0f;
        };

        // sliding W-length windows
        // rows
        for (int y = 0; y < N; y++) {
            for (int x = 0; x <= N - W; x++) {
                eval += eval_window(x, y, 1, 0);
            }
        }

        // cols
        for (int x = 0; x < N; x++) {
            for (int y = 0; y <= N - W; y++) {
                eval += eval_window(x, y, 0, 1);
            }
        }

        // diagonal TL -> BR
        for (int y = 0; y <= N - W; y++) {
            for (int x = 0; x <= N - W; x++) {
                eval += eval_window(x, y, 1, 1);
            }
        }

        // diagonal TR -> BL
        for (int y = 0; y <= N - W; y++) {
            for (int x = W - 1; x < N; x++) {
                eval += eval_window(x, y, -1, 1);
            }
        }

        return std::clamp(eval, -1.0f, 1.0f);
    }

    float minimax(TicTacToe<N, W>& game, float alpha, float beta, int depth, int prev_move = -1) {
        if (depth <= 0) return get_static_eval(game, prev_move);

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

template<int N, int W, int B = 1048576>
struct OldMinimaxAgent {
    int depth;
    std::string name;

    OldMinimaxAgent() : depth(9999), name("Unnamed MinimaxAgent") {}
    OldMinimaxAgent(int depth) : depth(depth), name("Unnamed MinimaxAgent") {}
    OldMinimaxAgent(int depth, std::string name) : depth(depth), name(name) {}

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
        auto check_direction = [&](int dx, int dy) -> int {
            // keep count of how many we actually have in a row
            // note how many we could get in a row in +dx +dy with one added
            // note how many we could get in a row in -dy -dy with one added

            int x = move_x + dx;
            int y = move_y + dy;

            int num_in_a_row = 1; // starts at 1 because of the move were checking
            int possible_pos_in_a_row = 0;
            int possible_neg_in_a_row = 0;

            while (x >= 0 && x < N && y >= 0 && y < N) {
                if (possible_pos_in_a_row == 0) {
                    if (game.at(x, y) == game.next_player) num_in_a_row++;
                    else if (game.at(x, y) == BoardSquare::EMPTY) possible_pos_in_a_row++;
                } else {
                    if (game.at(x, y) == game.next_player) possible_pos_in_a_row++;
                    else if (game.at(x, y) == BoardSquare::EMPTY) possible_pos_in_a_row++;
                }

                x += dx;
                y += dy;
            }

            x = move_x - dx;
            y = move_y - dy;

            while (x >= 0 && x < N && y >= 0 && y < N) {
                if (possible_neg_in_a_row == 0) {
                    if (game.at(x, y) == game.next_player) num_in_a_row++;
                    else if (game.at(x, y) == BoardSquare::EMPTY) possible_neg_in_a_row++;
                    else break;
                } else {
                    if (game.at(x, y) == game.next_player) possible_neg_in_a_row++;
                    else if (game.at(x, y) == BoardSquare::EMPTY) possible_neg_in_a_row++;
                    else break;
                }

                x -= dx;
                y -= dy;
            }

            if (num_in_a_row + possible_pos_in_a_row >= W && num_in_a_row + possible_neg_in_a_row >= W) return 2;
            else if (num_in_a_row + possible_pos_in_a_row >= W) return 1;
            else if (num_in_a_row + possible_neg_in_a_row >= W) return 1;
            else return 0;
        };

        int fork_count = 0;
        fork_count += check_direction(1, 0);
        fork_count += check_direction(0, 1);
        fork_count += check_direction(1, 1);
        fork_count += check_direction(-1, 1);

        if (fork_count >= 2) return 0.9f;
        if (fork_count == 1) return 0.5f;
        else return 0.0f;
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
        // 3) play towards the middle - 0.2f

        float heuristic = 0.0f;

        int move_x = move % N;
        int move_y = move / N;

        heuristic += block_heuristic(game, move_x, move_y, move);

        heuristic += win_heuristic(game, move);

        heuristic += fork_heuristic(game, move_x, move_y, move);

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

template<int N, int W, int S = 0>
struct RandomAgent {
    std::string name;
    RandomAgent() : name("Unnamed RandomAgent") {}
    RandomAgent(std::string name) : name(name) {}

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

    std::string& get_name() {
        return name;
    }
};

template<int N, int W, int S = 0>
struct HumanAgent {
    std::string name;

    HumanAgent() : name("Unnamed HuamnAgent") {}
    HumanAgent(std::string name) : name(name) {}

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

    std::string get_name() {
        return name;
    }
};

template<int N, int W, int S = 0>
struct FFNNAgent {
    std::string name;
    

    FFNNAgent() : name("Unnamed HuamnAgent") {}
    FFNNAgent(std::string name) : name(name) {}
};