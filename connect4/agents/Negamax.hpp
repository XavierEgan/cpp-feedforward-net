#pragma once
#include "../connect4.hpp"
#include "../enums.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

/*
flat two-tier transposition table keyed on the connect 4 perfect hash
(pos + mask). slot 0 is depth-preferred, slot 1 is always-replace — same
scheme as tictactoe's MinimaxRev4.

entries carry a generation number instead of being cleared between searches:
new_search() bumps the generation, which invalidates every old entry at zero
cost. per-move freshness matters — stale scores (with their depth-relative
win bonuses) from earlier root positions cause bad cutoffs, the same pathology
MinimaxRev4 fixed by clearing per move — but clearing tens of MB inside a
1ms move budget is a non-starter
*/
template<int TT_SIZE>
struct C4FlatTT {
    static_assert((TT_SIZE & (TT_SIZE - 1)) == 0, "TT_SIZE must be a power of two");

    struct Entry {
        uint64_t key = 0;
        float score = 0.0f;
        int8_t depth = -1;
        int8_t best_move = -1;
        Bound bound = Bound::EXACT;
        uint8_t gen = 0;
    };

    std::vector<Entry> data;
    uint8_t current_gen = 1; // entries default to gen 0, so they start invalid
    static constexpr uint64_t mask = TT_SIZE - 1;

    C4FlatTT() : data(TT_SIZE * 2) {}

    void clear() { std::fill(data.begin(), data.end(), Entry{}); }

    // invalidates all existing entries; call once per get_move
    void new_search() {
        current_gen++;
        if (current_gen == 0) {
            // generation counter wrapped — hard-clear so gen-0 entries can't
            // masquerade as fresh
            clear();
            current_gen = 1;
        }
    }

    // returns nullptr if not found
    const Entry* probe(uint64_t key) const {
        const size_t idx = static_cast<size_t>(key & mask) * 2;
        if (data[idx].key == key && data[idx].gen == current_gen) return &data[idx];
        if (data[idx + 1].key == key && data[idx + 1].gen == current_gen) return &data[idx + 1];
        return nullptr;
    }

    void store(uint64_t key, float score, int depth, Bound bound, int best_move) {
        const size_t idx = static_cast<size_t>(key & mask) * 2;
        Entry& depth_slot = data[idx];
        Entry& always_slot = data[idx + 1];

        Entry e;
        e.key = key;
        e.score = score;
        e.depth = static_cast<int8_t>(depth);
        e.best_move = static_cast<int8_t>(best_move);
        e.bound = bound;
        e.gen = current_gen;

        // stale-generation slots are free to overwrite
        if (depth_slot.gen != current_gen || depth_slot.key == key || depth >= depth_slot.depth) {
            depth_slot = e;
        } else {
            always_slot = e;
        }
    }
};

/*
iterative-deepening negamax with alpha-beta, a transposition table, and
connect-4-specific pruning:
    - immediate wins are taken without searching siblings
    - if the opponent has one playable winning spot, the blocking move is forced
    - if the opponent has two or more, the node is scored as lost immediately
    - moves are ordered TT-best first, then by threats created, then centre-out

leaf eval is the winning-spots differential (empty cells that would complete
4 for each side) plus a small centre-column bonus, clamped well inside the
win scores so search always prefers a real win
*/
template<int WIDTH, int HEIGHT, int TT_SIZE = (1 << 20)>
struct C4NegamaxAgent {
    std::chrono::duration<double, std::milli> max_move_time;
    std::string name;

    C4NegamaxAgent() : max_move_time(10.0), name("Unnamed C4NegamaxAgent") {}
    C4NegamaxAgent(double max_move_time_ms) : max_move_time(max_move_time_ms), name("Unnamed C4NegamaxAgent") {}
    C4NegamaxAgent(double max_move_time_ms, std::string n) : max_move_time(max_move_time_ms), name(std::move(n)) {}

    std::string& get_name() { return name; }

    int last_depth = -1;
    int get_last_depth() const { return last_depth; }

    // fixed-depth search score from RED's perspective
    float get_eval(Connect4<WIDTH, HEIGHT>& game) {
        tt.new_search();
        constexpr std::chrono::steady_clock::time_point far_future =
            std::chrono::steady_clock::time_point::max();
        const float stm_score = _negamax(game, -INF, INF, 8, far_future);
        return (game.next_player() == Disc::RED) ? stm_score : -stm_score;
    }

    int get_move(Connect4<WIDTH, HEIGHT>& game, unsigned int seed = std::random_device{}()) {
        // invalidate previous searches' entries (cheap generation bump) but
        // keep them across iterative-deepening depths within this move
        tt.new_search();

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

            // a forced win/loss is already found, deeper search changes nothing
            if (std::abs(_last_root_score) >= WIN_SCORE) break;
        }

        if (prev_move == -1) {
            // couldn't finish depth 1; emergency fallback
            for (int col = 0; col < WIDTH; col++)
                if (game.can_play(col)) return col;
        }
        return prev_move;
    }

private:
    using Game = Connect4<WIDTH, HEIGHT>;

    static constexpr float INF = 9999.0f;
    static constexpr float WIN_SCORE = 1.0f;
    static constexpr float WIN_EPS = 0.001f; // bonus per ply of remaining depth: prefer faster wins

    C4FlatTT<TT_SIZE> tt;
    float _last_root_score = 0.0f;

    // ── static evaluation, side-to-move perspective ──────────────────────────
    static float _leaf_eval(const Game& game) {
        const uint64_t me = game.pos;
        const uint64_t opp = game.pos ^ game.mask;

        const int my_spots = std::popcount(Game::winning_spots(me, game.mask));
        const int opp_spots = std::popcount(Game::winning_spots(opp, game.mask));

        const uint64_t centre = Game::column_mask(WIDTH / 2);
        const int my_centre = std::popcount(me & centre);
        const int opp_centre = std::popcount(opp & centre);

        const float eval = 0.04f * static_cast<float>(my_spots - opp_spots)
                         + 0.02f * static_cast<float>(my_centre - opp_centre);
        return std::clamp(eval, -0.9f, 0.9f);
    }

    // ── move ordering ────────────────────────────────────────────────────────
    // higher is better: TT best move, then threats created by the move, then
    // centre-out static preference
    static float _move_score(Game& game, int col, int tt_best_move) {
        if (col == tt_best_move) return 1000.0f;

        game.play_move(col);
        // after playing, pos is the opponent — our discs are pos ^ mask
        const int threats = std::popcount(Game::winning_spots(game.pos ^ game.mask, game.mask));
        game.unplay_move();

        const float centre = static_cast<float>(WIDTH / 2 - std::abs(col - WIDTH / 2));
        return static_cast<float>(threats) * 10.0f + centre;
    }

    // fills move_buf with {ordering score, column} sorted best-first, returns count
    static int _ordered_moves(Game& game, int tt_best_move, std::array<std::pair<float, int>, WIDTH>& move_buf) {
        int count = 0;
        for (int col = 0; col < WIDTH; col++) {
            if (!game.can_play(col)) continue;
            move_buf[count++] = {_move_score(game, col, tt_best_move), col};
        }
        std::sort(move_buf.begin(), move_buf.begin() + count,
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        return count;
    }

    /*
    tactical situation of the side to move, checked before searching children:
        win_col   – column that wins immediately (-1 if none)
        block_col – forced blocking column (-1 if none)
        lost      – opponent has 2+ playable wins and we can't win first
    */
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

    // ── negamax with alpha-beta + TT ─────────────────────────────────────────
    // returns score from the perspective of the side to move
    float _negamax(Game& game, float alpha, float beta, int depth,
                   std::chrono::steady_clock::time_point deadline) {
        if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

        if (game.is_full()) return 0.0f;

        const Tactics t = _tactics(game);
        if (t.win_col != -1) return WIN_SCORE + depth * WIN_EPS;
        if (t.lost) return -(WIN_SCORE + (depth - 1) * WIN_EPS);

        if (depth <= 0) return _leaf_eval(game);

        // forced block: the only move worth searching
        if (t.block_col != -1) {
            game.play_move(t.block_col);
            const float val = -_negamax(game, -beta, -alpha, depth - 1, deadline);
            game.unplay_move();
            return val;
        }

        // tt probe
        const uint64_t key = game.key();
        int tt_best = -1;
        {
            const auto* e = tt.probe(key);
            if (e) {
                tt_best = e->best_move;
                if (e->depth >= depth) {
                    const float s = e->score;
                    if (e->bound == Bound::EXACT) return s;
                    if (e->bound == Bound::LOWER) alpha = std::max(alpha, s);
                    else beta = std::min(beta, s);
                    if (alpha >= beta) return s;
                }
            }
        }

        std::array<std::pair<float, int>, WIDTH> moves;
        const int count = _ordered_moves(game, tt_best, moves);

        const float alpha_orig = alpha;
        const float beta_orig = beta;
        float best_score = -INF;
        int best_move = moves[0].second;

        for (int i = 0; i < count; i++) {
            game.play_move(moves[i].second);
            const float val = -_negamax(game, -beta, -alpha, depth - 1, deadline);
            game.unplay_move();

            // discard the result if the deadline hit mid-search — the 0.0f
            // return would corrupt alpha
            if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

            if (val > best_score) {
                best_score = val;
                best_move = moves[i].second;
            }
            alpha = std::max(alpha, val);
            if (alpha >= beta) break;
        }

        Bound bound = Bound::EXACT;
        if (best_score <= alpha_orig) bound = Bound::UPPER;
        else if (best_score >= beta_orig) bound = Bound::LOWER;
        tt.store(key, best_score, depth, bound, best_move);

        return best_score;
    }

    // ── root search for one iterative-deepening depth ────────────────────────
    // returns the best column, breaking ties randomly with the caller's seed
    int _root_search(Game& game, int depth,
                     std::chrono::steady_clock::time_point deadline, unsigned int seed) {
        const Tactics t = _tactics(game);
        if (t.win_col != -1) {
            _last_root_score = WIN_SCORE + depth * WIN_EPS;
            return t.win_col;
        }
        if (t.block_col != -1) {
            _last_root_score = 0.0f;
            return t.block_col;
        }

        int tt_best = -1;
        {
            const auto* e = tt.probe(game.key());
            if (e) tt_best = e->best_move;
        }

        std::array<std::pair<float, int>, WIDTH> moves;
        const int count = _ordered_moves(game, tt_best, moves);

        float alpha = -INF;
        float best_score = -INF;
        // {ordering score, column} of every move tied on search score. deep
        // searches prove many moves exactly equal (e.g. all drawn), so ties
        // are broken by the ordering heuristic first — picking uniformly at
        // random among "equal" moves throws away all positional pressure and
        // makes DEEPER search play WORSE
        std::vector<std::pair<float, int>> best_moves;

        for (int i = 0; i < count; i++) {
            game.play_move(moves[i].second);
            const float val = -_negamax(game, -INF, -alpha, depth - 1, deadline);
            game.unplay_move();

            if (std::chrono::steady_clock::now() >= deadline) break;

            if (val > best_score) {
                best_score = val;
                best_moves.clear();
                best_moves.push_back(moves[i]);
            } else if (val == best_score) {
                best_moves.push_back(moves[i]);
            }
            alpha = std::max(alpha, val);
        }

        _last_root_score = best_score;

        if (best_moves.empty()) return moves[0].second;

        // keep only the moves that also tie on the ordering heuristic, then
        // randomise among those for game variety
        const float best_order_score = std::max_element(best_moves.begin(), best_moves.end())->first;
        std::vector<int> candidates;
        for (const auto& [order_score, col] : best_moves)
            if (order_score == best_order_score) candidates.push_back(col);

        if (candidates.size() == 1) return candidates[0];

        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
        return candidates.at(dist(gen));
    }
};
