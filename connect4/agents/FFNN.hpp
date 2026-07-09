#pragma once
#include "../connect4.hpp"
#include "../enums.hpp"
#include "../../FFNN.hpp"
#include "../../Workspace.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <random>
#include <string>
#include <vector>

/*
uses the neural net as the leaf evaluator inside a shallow iterative-deepening
negamax. the net is trained to predict the game outcome from RED's perspective
(+1 red win, -1 yellow win), so leaf values are negated when yellow is to move.

immediate wins/forced blocks are still handled by exact tactics — the net only
has to judge quiet positions
*/
template<int WIDTH, int HEIGHT>
struct C4FFNNAgent {
    std::chrono::duration<double, std::milli> max_move_time;
    std::string name;
    FFNN ffnn;
    ForwardWorkspace ws; // reused across evals, allocation-free after the first call

    C4FFNNAgent(FFNN ffnn)
        : max_move_time(10.0), name("Unnamed C4FFNNAgent"), ffnn(std::move(ffnn)), ws(ForwardWorkspace::from_shape(this->ffnn.network_shape)) {}
    C4FFNNAgent(double max_move_time_ms, FFNN ffnn)
        : max_move_time(max_move_time_ms), name("Unnamed C4FFNNAgent"), ffnn(std::move(ffnn)), ws(ForwardWorkspace::from_shape(this->ffnn.network_shape)) {}
    C4FFNNAgent(double max_move_time_ms, std::string name, FFNN ffnn)
        : max_move_time(max_move_time_ms), name(std::move(name)), ffnn(std::move(ffnn)), ws(ForwardWorkspace::from_shape(this->ffnn.network_shape)) {}

    std::string& get_name() { return name; }

    int last_depth = -1;
    int get_last_depth() const { return last_depth; }

    // raw net prediction for the current position, from RED's perspective
    float get_eval(Connect4<WIDTH, HEIGHT>& game) {
        return ffnn.forward(game.get_board_state(), ws)(0, 0);
    }

    int get_move(Connect4<WIDTH, HEIGHT>& game, unsigned int seed = std::random_device{}()) {
        const std::chrono::steady_clock::time_point deadline =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(max_move_time);

        last_depth = -1;

        const int max_depth = Game::num_cells - game.num_moves;
        int prev_move = -1;

        for (int depth = 1; depth <= max_depth; depth++) {
            const int cur_move = _root_search(game, depth, deadline, seed);
            if (std::chrono::steady_clock::now() >= deadline) break;
            prev_move = cur_move;
            last_depth = depth;
        }

        if (prev_move == -1) {
            for (int col = 0; col < WIDTH; col++)
                if (game.can_play(col)) return col;
        }
        return prev_move;
    }

private:
    using Game = Connect4<WIDTH, HEIGHT>;

    static constexpr float INF = 9999.0f;
    static constexpr float WIN_SCORE = 1.0f;
    static constexpr float WIN_EPS = 0.001f;

    // net leaf eval converted to side-to-move perspective and kept inside the
    // win scores
    float _leaf_eval(Game& game) {
        const float red_eval = std::clamp(get_eval(game), -0.95f, 0.95f);
        return (game.next_player() == Disc::RED) ? red_eval : -red_eval;
    }

    struct Tactics {
        int win_col = -1;
        int block_col = -1;
        bool lost = false;
    };

    static Tactics _tactics(const Game& game) {
        Tactics t;
        const uint64_t possible = game.possible_moves();

        const uint64_t my_wins = Game::winning_spots(game.pos, game.mask) & possible;
        if (my_wins != 0) {
            t.win_col = std::countr_zero(my_wins) / Game::col_bits;
            return t;
        }

        const uint64_t opp_wins = Game::winning_spots(game.pos ^ game.mask, game.mask) & possible;
        if (std::popcount(opp_wins) >= 2) {
            t.lost = true;
        } else if (opp_wins != 0) {
            t.block_col = std::countr_zero(opp_wins) / Game::col_bits;
        }
        return t;
    }

    float _negamax(Game& game, float alpha, float beta, int depth,
                   std::chrono::steady_clock::time_point deadline) {
        if (std::chrono::steady_clock::now() >= deadline) return 0.0f;
        if (game.is_full()) return 0.0f;

        const Tactics t = _tactics(game);
        if (t.win_col != -1) return WIN_SCORE + depth * WIN_EPS;
        if (t.lost) return -(WIN_SCORE + (depth - 1) * WIN_EPS);

        if (depth <= 0) return _leaf_eval(game);

        if (t.block_col != -1) {
            game.play_move(t.block_col);
            const float val = -_negamax(game, -beta, -alpha, depth - 1, deadline);
            game.unplay_move();
            return val;
        }

        float best_score = -INF;

        // centre-out column order
        for (int i = 0; i < WIDTH; i++) {
            const int col = WIDTH / 2 + (i % 2 == 0 ? i / 2 : -(i / 2 + 1));
            if (!game.can_play(col)) continue;

            game.play_move(col);
            const float val = -_negamax(game, -beta, -alpha, depth - 1, deadline);
            game.unplay_move();

            if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

            best_score = std::max(best_score, val);
            alpha = std::max(alpha, val);
            if (alpha >= beta) break;
        }

        return best_score;
    }

    int _root_search(Game& game, int depth,
                     std::chrono::steady_clock::time_point deadline, unsigned int seed) {
        const Tactics t = _tactics(game);
        if (t.win_col != -1) return t.win_col;
        if (t.block_col != -1) return t.block_col;

        float alpha = -INF;
        float best_score = -INF;
        std::vector<int> best_moves;

        for (int i = 0; i < WIDTH; i++) {
            const int col = WIDTH / 2 + (i % 2 == 0 ? i / 2 : -(i / 2 + 1));
            if (!game.can_play(col)) continue;

            game.play_move(col);
            const float val = -_negamax(game, -INF, -alpha, depth - 1, deadline);
            game.unplay_move();

            if (std::chrono::steady_clock::now() >= deadline) break;

            if (val > best_score) {
                best_score = val;
                best_moves.clear();
                best_moves.push_back(col);
            } else if (val == best_score) {
                best_moves.push_back(col);
            }
            alpha = std::max(alpha, val);
        }

        if (best_moves.empty()) {
            for (int col = 0; col < WIDTH; col++)
                if (game.can_play(col)) return col;
            return -1;
        }
        if (best_moves.size() == 1) return best_moves[0];

        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(0, static_cast<int>(best_moves.size()) - 1);
        return best_moves.at(dist(gen));
    }
};
