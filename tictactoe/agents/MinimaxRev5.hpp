#pragma once
#include "../tictactoe.hpp"
#include "../enums.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Build (with OpenMP): g++ -std=c++23 -O3 -fopenmp -o train tictactoe/train.cpp
// Build (without):     g++ -std=c++23 -O3            -o train tictactoe/train.cpp

// ─────────────────────────────────────────────────────────────────────────────
//  Rev5Config — compile-time feature toggles (C++23 structural non-type template)
//
//  All fields default to true (all features on).  To disable a feature, pass
//  a custom Rev5Config as the CFG template argument:
//
//      MinimaxRev5Agent<3, 3, (1<<20), Rev5Config{.use_qsearch = false}> agent;
//
//  Toggleable features:
//    use_tt           – Transposition table (probe + store)
//    use_pvs          – Principal Variation Search / Negascout
//    use_qsearch      – Quiescence search (depth-0 tactical extension)
//    use_killers      – Killer move heuristic (2 slots per ply)
//    use_history      – History heuristic (per-player, per-cell)
//    use_win_pruning  – Immediate win / forced-block detection before full search
//    use_futility     – Futility pruning at depth == 1
//    use_move_order   – Tactical move ordering (win/block/fork/center/history blend)
//    use_w2_threats   – W-2 threat counting in leaf evaluation
//    use_threat1_eval – W-1 threat weight in leaf evaluation
//    use_omp          – OpenMP root-level parallelism (requires -fopenmp to speedup)
//
//  Always active (not toggleable): alpha-beta pruning, IDDFS.
// ─────────────────────────────────────────────────────────────────────────────
struct Rev5Config {
    bool use_tt           = true;
    bool use_pvs          = true;
    bool use_qsearch      = true;
    bool use_killers      = true;
    bool use_history      = true;
    bool use_win_pruning  = true;
    bool use_futility     = true;
    bool use_move_order   = true;
    bool use_w2_threats   = true;
    bool use_threat1_eval = true;
    bool use_omp          = true;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Thread-safe flat two-tier transposition table
//
//  Lockless design: the key and payload are decoupled into separate arrays.
//  Writes use a two-phase protocol:
//    1. Write payload fields (no ordering constraint — key is still stale, so
//       no reader can match this slot during Phase 1).
//    2. Publish key with memory_order_release.
//  Readers acquire-load the key first, then read the payload — the acquire/
//  release pair guarantees the reader sees fully written payload whenever it
//  sees the matching key.
//
//  Concurrent writes to the same slot produce benign data races on the payload
//  (heuristic values only).  This is standard practice in parallel game-tree
//  search — see Hyatt & Marsland, "A Study of Parallel Search Algorithms".
//
//  Slot selection:
//    slot 0 – depth-preferred  (replaced only if incoming depth >= stored depth)
//    slot 1 – always-replace   (spillover for lower-depth, or same-key updates)
// ─────────────────────────────────────────────────────────────────────────────
template<int TT_SIZE>
struct Rev5FlatTT {
    static_assert((TT_SIZE & (TT_SIZE - 1)) == 0, "TT_SIZE must be a power of two");

    static constexpr int SLOTS = TT_SIZE * 2;
    static constexpr int mask  = TT_SIZE - 1;

    struct Payload {
        float    score     = 0.0f;
        int8_t   depth     = -1;
        int8_t   best_move = -1;
        NodeType type      = NodeType::EXACT;
    };

    struct ProbeResult {
        float    score;
        int      depth;
        NodeType type;
        int      best_move;
        bool     hit;
    };

    // Heap-allocated to avoid stack overflow for large TT_SIZE.
    std::vector<Payload>                data; // non-atomic payload
    std::vector<std::atomic<long long>> keys; // atomic keys  (0 = empty slot)

    Rev5FlatTT() : data(SLOTS), keys(SLOTS) {
        for (auto& k : keys) k.store(0LL, std::memory_order_relaxed);
    }

    // Not thread-safe — call only in single-threaded phases (before/after search).
    void clear() {
        std::fill(data.begin(), data.end(), Payload{});
        for (auto& k : keys) k.store(0LL, std::memory_order_relaxed);
    }

    // Thread-safe probe.  Returns hit=false if the key is absent or the slot is invalid.
    ProbeResult probe(long long key) const {
        int idx = static_cast<int>(static_cast<unsigned long long>(key) & mask) * 2;
        for (int s = 0; s < 2; ++s) {
            // Acquire-load: guarantees we see the payload written before this key was published.
            long long k = keys[idx + s].load(std::memory_order_acquire);
            if (k != key) continue;
            const Payload& p = data[idx + s];
            if (p.depth < 0) continue; // empty or invalid slot
            return { p.score, p.depth, p.type, p.best_move, true };
        }
        return { 0.0f, -1, NodeType::EXACT, -1, false };
    }

    // Thread-safe store (lockless two-phase write: payload first, then key with release).
    void store(long long key, float score, int depth, NodeType type, int best_move) {
        int idx = static_cast<int>(static_cast<unsigned long long>(key) & mask) * 2;

        // Choose slot: prefer updating existing entry; otherwise depth-preferred vs spill.
        int slot = 1;
        {
            long long k0 = keys[idx].load(std::memory_order_relaxed);
            if (k0 == key) {
                slot = 0;
            } else {
                int stored_depth = (k0 != 0LL) ? static_cast<int>(data[idx].depth) : -1;
                if (stored_depth < 0 || depth >= stored_depth) slot = 0;
            }
        }

        Payload p;
        p.score     = score;
        p.depth     = static_cast<int8_t>(depth);
        p.best_move = static_cast<int8_t>(best_move);
        p.type      = type;

        // Phase 1: write payload (reader can't match yet — key is still old).
        data[idx + slot] = p;
        // Phase 2: publish key with release — reader acquire-loads key and then
        //          is guaranteed to see the payload written above.
        keys[idx + slot].store(key, std::memory_order_release);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  MinimaxRev5Agent — Rev4 with compile-time feature toggles + OMP root parallelism
//
//  Template parameters:
//    N   – board side length
//    W   – win length
//    TS  – TT size in buckets (power of 2; default ≈ 4 M entries)
//    CFG – Rev5Config controlling which features are compiled in (default: all on)
//
//  OpenMP note:
//    When CFG.use_omp = true, each IDDFS iteration searches root moves in parallel.
//    Each thread operates on its own copy of the game state.  The transposition
//    table is shared (lockless writes; see Rev5FlatTT).  killers/history are
//    shared member arrays; concurrent accesses produce benign data races — both
//    are heuristic structures where occasional stale/torn reads are harmless.
//    PVS first-move ordering is relaxed at the root level (all root moves are
//    searched with a full window to remove inter-thread ordering dependencies).
//    PVS still applies correctly inside each thread's recursive _negamax calls.
//
//  If '-fopenmp' is not passed to the compiler, OMP pragmas are silently ignored
//  and the agent runs single-threaded regardless of CFG.use_omp.
// ─────────────────────────────────────────────────────────────────────────────
template<int N, int W, int TS = (1 << 22), Rev5Config CFG = Rev5Config{}>
struct MinimaxRev5Agent {
    std::chrono::duration<double, std::milli> max_move_time;
    std::string name;

    MinimaxRev5Agent()
        : max_move_time(1.0), name("Unnamed MinimaxRev5Agent") { _init(); }
    MinimaxRev5Agent(double max_move_time_ms)
        : max_move_time(max_move_time_ms), name("Unnamed MinimaxRev5Agent") { _init(); }
    MinimaxRev5Agent(double max_move_time_ms, std::string n)
        : max_move_time(max_move_time_ms), name(std::move(n)) { _init(); }

    std::string& get_name() { return name; }

    int last_depth = -1;

    int get_last_depth() const { return last_depth; }

    // Returns a static evaluation for the current position.
    float get_eval(TicTacToe<N, W>& game) {
        if constexpr (CFG.use_tt) tt.clear();
        _clear_history();
        _clear_killers();
        constexpr std::chrono::steady_clock::time_point far_future =
            std::chrono::steady_clock::time_point::max();
        return _negamax(game, -INF, INF, 4, far_future, -1, 0);
    }

    int get_move(TicTacToe<N, W>& game, int seed = std::random_device{}()) {
        std::chrono::steady_clock::time_point deadline =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(max_move_time);

        _clear_history();
        _clear_killers();
        if constexpr (CFG.use_tt) tt.clear();

        int prev_move = -1;

        for (int depth = 0; ; ++depth) {
            auto [m, s] = _root_search(game, -INF, INF, depth, deadline, seed);

            if (std::chrono::steady_clock::now() >= deadline) {
                if (prev_move == -1) {
                    // Couldn't finish even depth 0; emergency fallback.
                    for (int mv = 0; mv < N * N; ++mv)
                        if (game.at(mv) == BoardSquare::EMPTY) return mv;
                }
                return prev_move;
            }

            if (m != -1) prev_move = m;
            last_depth = depth;
            if (depth >= N * N) break;
        }
        return prev_move;
    }

private:
    // ── Constants ────────────────────────────────────────────────────────────
    static constexpr float INF              = 9999.0f;
    static constexpr float WIN_SCORE        = 1.0f;
    static constexpr float WIN_EPS          = 0.001f;
    static constexpr float FUTILITY_MARGIN  = 0.5f;
    static constexpr int   MAX_DEPTH        = N * N;
    static constexpr int   NUM_KILLERS      = 2;
    static constexpr int   QSEARCH_MAX_DEPTH = 2;

    // ── Data ─────────────────────────────────────────────────────────────────
    Rev5FlatTT<TS> tt;

    // History table: history[player_idx][cell]  (0=X, 1=O)
    std::array<std::array<int32_t, N * N>, 2> history{};

    // Killer moves: killers[ply][slot]
    std::array<std::array<int8_t, NUM_KILLERS>, MAX_DEPTH + 1> killers{};

    void _init() {
        _clear_history();
        _clear_killers();
    }

    void _clear_history() { for (auto& row : history) row.fill(0); }
    void _clear_killers() { for (auto& row : killers) row.fill(-1); }

    void _store_killer(int ply, int move) {
        if constexpr (!CFG.use_killers) return;
        if (ply < 0 || ply > MAX_DEPTH) return;
        auto& ks = killers[ply];
        if (ks[0] == move) return;
        ks[1] = ks[0];
        ks[0] = static_cast<int8_t>(move);
    }

    bool _is_killer(int ply, int move) const {
        if constexpr (!CFG.use_killers) return false;
        if (ply < 0 || ply > MAX_DEPTH) return false;
        const auto& ks = killers[ply];
        return ks[0] == move || ks[1] == move;
    }

    // ── Move ordering ─────────────────────────────────────────────────────────
    // Returns a priority score (higher = search first).
    // When use_move_order = false, all moves get 0.0f (order is arbitrary).
    //
    // Priorities (descending) when use_move_order = true:
    //   5000 : TT best move
    //   4000 : immediate win
    //   3000 : immediate block (opponent W-1 threat)
    //   2000 : fork (≥2 new W-1 threats)
    //   1500 : killer move
    //   1000 : single threat
    //    500 : center proximity + history blend
    float _move_score(TicTacToe<N, W>& game, int move, int tt_best_move, int ply) const {
        if constexpr (!CFG.use_move_order) return 0.0f;

        if (move == tt_best_move) return 5000.0f;

        bool is_x = (game.next_player == BoardSquare::X);
        int win_moves = 0, block_moves = 0, new_threats = 0;

        for (int wi : game.cell_to_windows[move]) {
            const auto& ws = game.window_states[wi];
            if (is_x) {
                if (ws.num_x == W - 1 && ws.num_o == 0) ++win_moves;
                if (ws.num_o == W - 1 && ws.num_x == 0) ++block_moves;
                if constexpr (W >= 3)
                    if (ws.num_x == W - 2 && ws.num_o == 0) ++new_threats;
            } else {
                if (ws.num_o == W - 1 && ws.num_x == 0) ++win_moves;
                if (ws.num_x == W - 1 && ws.num_o == 0) ++block_moves;
                if constexpr (W >= 3)
                    if (ws.num_o == W - 2 && ws.num_x == 0) ++new_threats;
            }
        }

        if (win_moves  > 0) return 4000.0f + win_moves;
        if (block_moves > 0) return 3000.0f + block_moves;
        if (new_threats >= 2) return 2000.0f + new_threats;
        if (_is_killer(ply, move)) return 1500.0f;
        if (new_threats == 1) return 1000.0f;

        float cx  = (N - 1) * 0.5f;
        float cy  = (N - 1) * 0.5f;
        float mx  = static_cast<float>(move % N);
        float my  = static_cast<float>(move / N);  // NOLINT: intentional integer division before cast
        float dist     = std::abs(mx - cx) + std::abs(my - cy);
        float max_dist = cx + cy;
        float center   = (max_dist > 0.0f) ? (1.0f - dist / max_dist) : 1.0f;

        if constexpr (CFG.use_history) {
            int32_t hist = history[is_x ? 0 : 1][move];
            return 500.0f + center * 10.0f + std::log1p(static_cast<float>(hist));
        } else {
            return 500.0f + center * 10.0f;
        }
    }

    // ── Static evaluation ─────────────────────────────────────────────────────
    // Returns score from the perspective of the side to move (positive = good).
    float _leaf_eval(TicTacToe<N, W>& game, int prev_move) const {
        if (prev_move != -1) {
            BoardSquare w = game.check_winner(prev_move);
            if (w != BoardSquare::EMPTY) return -(WIN_SCORE);
        }

        float eval = std::clamp(game.static_eval, -1.0f, 1.0f);

        if constexpr (CFG.use_threat1_eval) {
            constexpr float THREAT_WEIGHT = 0.05f;
            float threat_diff = static_cast<float>(game.x_threat_count - game.o_threat_count) * THREAT_WEIGHT;
            eval = std::clamp(eval + threat_diff, -1.0f, 1.0f);
        }

        if constexpr (CFG.use_w2_threats && W >= 3) {
            constexpr float THREAT2_WEIGHT = 0.015f;
            int x_threats2 = 0, o_threats2 = 0;
            for (const auto& ws : game.window_states) {
                if (ws.num_x == W - 2 && ws.num_o == 0) ++x_threats2;
                if (ws.num_o == W - 2 && ws.num_x == 0) ++o_threats2;
            }
            float threat2_diff = static_cast<float>(x_threats2 - o_threats2) * THREAT2_WEIGHT;
            eval = std::clamp(eval + threat2_diff, -1.0f, 1.0f);
        }

        return (game.next_player == BoardSquare::X) ? eval : -eval;
    }

    // ── Immediate win detection ───────────────────────────────────────────────
    // Returns the winning move if the side to move can win in 1, else -1.
    int _find_immediate_win(TicTacToe<N, W>& game) const {
        bool is_x = (game.next_player == BoardSquare::X);
        for (int move = 0; move < N * N; ++move) {
            if (game.at(move) != BoardSquare::EMPTY) continue;
            for (int wi : game.cell_to_windows[move]) {
                const auto& ws = game.window_states[wi];
                if (is_x  && ws.num_x == W - 1 && ws.num_o == 0) return move;
                if (!is_x && ws.num_o == W - 1 && ws.num_x == 0) return move;
            }
        }
        return -1;
    }

    // ── Quiescence search ─────────────────────────────────────────────────────
    // Called at depth <= 0.  Keeps searching while either side has W-1 threats,
    // up to QSEARCH_MAX_DEPTH extra plies.  Uses a stand-pat lower bound.
    float _qsearch(TicTacToe<N, W>& game, float alpha, float beta,
                   std::chrono::steady_clock::time_point deadline, int prev_move,
                   int ply, int qdepth = 0) {
        if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

        if (prev_move != -1) {
            BoardSquare w = game.check_winner(prev_move);
            if (w != BoardSquare::EMPTY) return -(WIN_SCORE);
        }

        float stand_pat = _leaf_eval(game, -1);
        if (stand_pat >= beta) return stand_pat;
        if (stand_pat > alpha) alpha = stand_pat;

        bool any_threat = (game.x_threat_count > 0 || game.o_threat_count > 0);
        if (!any_threat || qdepth >= QSEARCH_MAX_DEPTH) return stand_pat;

        if constexpr (CFG.use_win_pruning) {
            int win_move = _find_immediate_win(game);
            if (win_move != -1) return WIN_SCORE;
        }

        bool  is_x = (game.next_player == BoardSquare::X);
        float best = stand_pat;

        for (int mv = 0; mv < N * N; ++mv) {
            if (game.at(mv) != BoardSquare::EMPTY) continue;

            bool tactical = false;
            for (int wi : game.cell_to_windows[mv]) {
                const auto& ws = game.window_states[wi];
                if (is_x) {
                    if ((ws.num_x == W - 1 && ws.num_o == 0) ||
                        (ws.num_o == W - 1 && ws.num_x == 0) ||
                        (W >= 3 && ws.num_x == W - 2 && ws.num_o == 0)) {
                        tactical = true; break;
                    }
                } else {
                    if ((ws.num_o == W - 1 && ws.num_x == 0) ||
                        (ws.num_x == W - 1 && ws.num_o == 0) ||
                        (W >= 3 && ws.num_o == W - 2 && ws.num_x == 0)) {
                        tactical = true; break;
                    }
                }
            }
            if (!tactical) continue;

            game.play_move(mv);
            float val = -_qsearch(game, -beta, -alpha, deadline, mv, ply + 1, qdepth + 1);
            game.unplay_move(mv);

            if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

            if (val > best)  best  = val;
            if (val > alpha) alpha = val;
            if (alpha >= beta) break;
        }

        return best;
    }

    // ── Negamax (alpha-beta + optional PVS + all pruning toggles) ────────────
    // Returns score from the perspective of the side to move (positive = good for mover).
    // `ply` is distance from root (0 at root), used for killer table indexing.
    float _negamax(TicTacToe<N, W>& game, float alpha, float beta, int depth,
                   std::chrono::steady_clock::time_point deadline, int prev_move, int ply) {
        if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

        // ── Terminal: winner check ────────────────────────────────────────────
        if (prev_move != -1) {
            BoardSquare w = game.check_winner(prev_move);
            if (w != BoardSquare::EMPTY)
                return -(WIN_SCORE + depth * WIN_EPS);
        }

        // ── Leaf node ─────────────────────────────────────────────────────────
        if (depth <= 0) {
            if constexpr (CFG.use_qsearch)
                return _qsearch(game, alpha, beta, deadline, prev_move, ply);
            else
                return _leaf_eval(game, prev_move);
        }

        // ── TT probe ─────────────────────────────────────────────────────────
        long long key     = game.hash_val;
        int       tt_best = -1;
        if constexpr (CFG.use_tt) {
            auto r = tt.probe(key);
            if (r.hit) {
                tt_best = r.best_move;
                if (r.depth >= depth) {
                    float s = r.score;
                    if (r.type == NodeType::EXACT) return s;
                    if (r.type == NodeType::LOWER) alpha = std::max(alpha, s);
                    else                           beta  = std::min(beta,  s);
                    if (alpha >= beta) return s;
                }
            }
        }

        // ── Immediate win pruning ─────────────────────────────────────────────
        if constexpr (CFG.use_win_pruning) {
            int win_move = _find_immediate_win(game);
            if (win_move != -1) {
                float score = WIN_SCORE + depth * WIN_EPS;
                if constexpr (CFG.use_tt)
                    tt.store(key, score, depth, NodeType::EXACT, win_move);
                return score;
            }
        }

        // ── Futility pruning (depth == 1) ──────────────────────────────────── 
        bool futility_active = false;
        if constexpr (CFG.use_futility) {
            if (depth == 1) {
                float futility_base = _leaf_eval(game, -1);
                futility_active = (futility_base + FUTILITY_MARGIN < alpha);
            }
        }

        // ── Generate and sort moves ───────────────────────────────────────────
        std::array<std::pair<float, int>, N * N> move_buf;
        int count = 0;
        for (int mv = 0; mv < N * N; ++mv) {
            if (game.at(mv) != BoardSquare::EMPTY) continue;
            move_buf[count++] = {_move_score(game, mv, tt_best, ply), mv};
        }

        if (count == 0) return 0.0f; // board full → draw

        std::sort(move_buf.begin(), move_buf.begin() + count,
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // ── Alpha-beta loop ───────────────────────────────────────────────────
        const float alpha_orig = alpha;
        const float beta_orig  = beta;
        float best_score = -INF;
        int   best_move  = move_buf[0].second;

        for (int i = 0; i < count; ++i) {
            if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

            int mv = move_buf[i].second;

            // Futility: skip quiet moves at depth 1 when enabled.
            if constexpr (CFG.use_futility) {
                if (futility_active && move_buf[i].first < 1000.0f) continue;
            }

            game.play_move(mv);

            float val;
            if constexpr (CFG.use_pvs) {
                if (i == 0) {
                    val = -_negamax(game, -beta, -alpha, depth - 1, deadline, mv, ply + 1);
                } else {
                    val = -_negamax(game, -alpha - 1e-4f, -alpha, depth - 1, deadline, mv, ply + 1);
                    if (val > alpha && std::chrono::steady_clock::now() < deadline)
                        val = -_negamax(game, -beta, -alpha, depth - 1, deadline, mv, ply + 1);
                }
            } else {
                val = -_negamax(game, -beta, -alpha, depth - 1, deadline, mv, ply + 1);
            }

            game.unplay_move(mv);

            // Discard result if deadline was hit — 0.0f would corrupt alpha.
            if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

            if (val > best_score) {
                best_score = val;
                best_move  = mv;
            }
            alpha = std::max(alpha, val);
            if (alpha >= beta) {
                _store_killer(ply, mv);
                if constexpr (CFG.use_history)
                    history[game.next_player == BoardSquare::X ? 0 : 1][mv] += depth * depth;
                break;
            }
        }

        // ── TT store ─────────────────────────────────────────────────────────
        if constexpr (CFG.use_tt) {
            if (std::chrono::steady_clock::now() < deadline) {
                NodeType type = NodeType::EXACT;
                if      (best_score <= alpha_orig) type = NodeType::UPPER;
                else if (best_score >= beta_orig)  type = NodeType::LOWER;
                tt.store(key, best_score, depth, type, best_move);
            }
        }

        return best_score;
    }

    // ── Root search for one IDDFS depth ──────────────────────────────────────
    // Returns {best_move, best_score} in the mover's frame.
    std::pair<int, float> _root_search(TicTacToe<N, W>& game,
                                       float alpha, float beta,
                                       int depth,
                                       std::chrono::steady_clock::time_point deadline,
                                       int seed) {
        // TT hint for move ordering from the previous IDDFS iteration.
        int tt_best = -1;
        if constexpr (CFG.use_tt) {
            auto r = tt.probe(game.hash_val);
            if (r.hit) tt_best = r.best_move;
        }

        // Generate and sort moves.
        std::array<std::pair<float, int>, N * N> move_buf;
        int count = 0;
        for (int mv = 0; mv < N * N; ++mv) {
            if (game.at(mv) != BoardSquare::EMPTY) continue;
            move_buf[count++] = {_move_score(game, mv, tt_best, 0), mv};
        }

        if (count == 0) return {-1, 0.0f};

        std::sort(move_buf.begin(), move_buf.begin() + count,
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Depth 0: just return the best-ordered move.
        if (depth == 0)
            return {move_buf[0].second, _move_score(game, move_buf[0].second, tt_best, 0)};

        float best_score = -INF;
        int   best_move  = move_buf[0].second;
        std::vector<int> best_moves;

        const float alpha_orig = alpha;
        const float beta_orig  = beta;

        if constexpr (CFG.use_omp) {
            // ── Parallel root search ──────────────────────────────────────────
            //
            // Each thread receives its own copy of game state (TicTacToe is copyable)
            // and independently searches one root move.
            //
            // TT is shared (see Rev5FlatTT lockless protocol).
            //
            // killers/history are shared member arrays.  Concurrent accesses produce
            // benign data races: both are purely heuristic so occasional torn/stale
            // reads only affect move ordering quality, not correctness.
            //
            // PVS is relaxed at root level: all root moves use a full window to
            // eliminate inter-thread ordering dependencies.  PVS applies normally
            // within each thread's recursive _negamax calls.
            //
            // alpha is shared and updated inside a critical section so threads
            // launched after a strong early result benefit from a tighter window.
            float shared_alpha = alpha;

            #pragma omp parallel for schedule(dynamic, 1) \
                shared(best_score, best_move, best_moves, shared_alpha)
            for (int i = 0; i < count; ++i) {
                // Read current best alpha (may have been tightened by an earlier thread).
                float cur_alpha;
                #pragma omp critical (rev5_root_alpha)
                { cur_alpha = shared_alpha; }

                if (cur_alpha >= beta) continue; // already cut off by another thread
                if (std::chrono::steady_clock::now() >= deadline) continue;

                int mv = move_buf[i].second;

                // Thread-local game copy — safe to mutate without affecting other threads.
                TicTacToe<N, W> local_game = game;
                local_game.play_move(mv);

                // Full-window search (PVS ordering relaxed at root for thread safety).
                float val = -_negamax(local_game, -beta, -cur_alpha, depth - 1, deadline, mv, 1);

                if (std::chrono::steady_clock::now() >= deadline) continue;

                #pragma omp critical (rev5_root_alpha)
                {
                    if (val > best_score) {
                        best_score = val;
                        best_move  = mv;
                        best_moves.clear();
                        best_moves.push_back(mv);
                        if (val > shared_alpha) shared_alpha = val;
                    } else if (val == best_score) {
                        best_moves.push_back(mv);
                    }
                }
            }

        } else {
            // ── Sequential root search ────────────────────────────────────────
            for (int i = 0; i < count; ++i) {
                if (std::chrono::steady_clock::now() >= deadline) break;

                int mv = move_buf[i].second;

                game.play_move(mv);

                float val;
                if constexpr (CFG.use_pvs) {
                    if (i == 0) {
                        val = -_negamax(game, -beta, -alpha, depth - 1, deadline, mv, 1);
                    } else {
                        val = -_negamax(game, -alpha - 1e-4f, -alpha, depth - 1, deadline, mv, 1);
                        if (val > alpha && val < beta && std::chrono::steady_clock::now() < deadline)
                            val = -_negamax(game, -beta, -alpha, depth - 1, deadline, mv, 1);
                    }
                } else {
                    val = -_negamax(game, -beta, -alpha, depth - 1, deadline, mv, 1);
                }

                game.unplay_move(mv);

                if (std::chrono::steady_clock::now() >= deadline) break;

                if (val > best_score) {
                    best_score = val;
                    best_move  = mv;
                    best_moves.clear();
                    best_moves.push_back(mv);
                } else if (val == best_score) {
                    best_moves.push_back(mv);
                }

                alpha = std::max(alpha, val);
                if (alpha >= beta) break;
            }
        }

        // Random tie-break among equal-scoring moves.
        if (best_moves.size() > 1) {
            std::mt19937 gen(static_cast<unsigned>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
            std::uniform_int_distribution<int> dist(0, static_cast<int>(best_moves.size()) - 1);
            best_move = best_moves[dist(gen)];
        }

        // Store root result in TT.
        if constexpr (CFG.use_tt) {
            if (std::chrono::steady_clock::now() < deadline) {
                NodeType type = NodeType::EXACT;
                if      (best_score <= alpha_orig) type = NodeType::UPPER;
                else if (best_score >= beta_orig)  type = NodeType::LOWER;
                tt.store(game.hash_val, best_score, depth, type, best_move);
            }
        }

        return {best_move, best_score};
    }
};
