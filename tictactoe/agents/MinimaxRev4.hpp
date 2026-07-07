#pragma once
#include "../tictactoe.hpp"
#include "../enums.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// I pretty much told an ai model to improve Rev3 and ended up with this

// ─────────────────────────────────────────────────────────────────────────────
//  Flat two-tier Transposition Table
//
//  TT_SIZE must be a power of 2.  Each bucket index holds two slots:
//    slot 0  – depth-preferred  (replaced only if incoming depth >= stored depth)
//    slot 1  – always-replace   (always replaced; acts as a spillover)
//
//  Stores the best move for each entry so root/PVS can reorder children.
// ─────────────────────────────────────────────────────────────────────────────
template<int TT_SIZE>
struct FlatTT {
    static_assert((TT_SIZE & (TT_SIZE - 1)) == 0, "TT_SIZE must be a power of two");

    struct Entry {
        long long key   = 0;
        float     score = 0.0f;
        int8_t    depth = -1;
        int8_t    best_move = -1;
        NodeType  type  = NodeType::EXACT;
    };

    // Two slots per bucket index → 2 * TT_SIZE entries total.
    // Heap-allocated to avoid stack overflow for large TT_SIZE values.
    std::vector<Entry> data;

    static constexpr int mask = TT_SIZE - 1;

    FlatTT() : data(TT_SIZE * 2) {}

    void clear() { std::fill(data.begin(), data.end(), Entry{}); }

    // Returns nullptr if not found.
    const Entry* probe(long long key) const {
        int idx = static_cast<int>(static_cast<unsigned long long>(key) & mask) * 2;
        if (data[idx    ].key == key && data[idx    ].depth >= 0) return &data[idx    ];
        if (data[idx + 1].key == key && data[idx + 1].depth >= 0) return &data[idx + 1];
        return nullptr;
    }

    void store(long long key, float score, int depth, NodeType type, int best_move) {
        int idx = static_cast<int>(static_cast<unsigned long long>(key) & mask) * 2;
        Entry& depth_slot  = data[idx    ];
        Entry& always_slot = data[idx + 1];

        Entry e;
        e.key       = key;
        e.score     = score;
        e.depth     = static_cast<int8_t>(depth);
        e.best_move = static_cast<int8_t>(best_move);
        e.type      = type;

        // Depth-preferred slot: replace if same key, or if incoming depth beats stored.
        if (depth_slot.depth < 0 || depth_slot.key == key || depth >= depth_slot.depth) {
            depth_slot = e;
        } else {
            // Otherwise spill to always-replace slot.
            always_slot = e;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  MinimaxRev4 — Optimised alpha-beta minimax agent
//
//  Improvements over Rev3:
//    1.  Flat two-tier transposition table (no map overhead, no full clears).
//    2.  TT not cleared between IDDFS depths → best moves carry forward.
//    3.  TT best-move used as first candidate in move ordering.
//    4.  Principal Variation Search (PVS / Negascout).
//    5.  Killer move heuristic (2 slots per ply).
//    6.  History heuristic table (per-player, per-cell).
//    7.  Immediate win / forced-block detection before full move generation.
//    8.  Futility pruning at depth 1 nodes.
//    9.  Improved leaf eval using W-1 and W-2 threat-count differential.
//   10.  Quiescence search — at depth=0, keep searching while W-1 threats exist (cap=2).
//   11.  Timeout guard after each recursive call to prevent alpha corruption.
//   12.  TT cleared per move to prevent stale scores from earlier positions causing bad cutoffs.
//
//  Note: LMR was intentionally removed — this game is too tactical for late-move
//  reductions to be safe (quiet moves can be critical for threat/block chains).
//
//  Template parameters:
//    N   – board side length
//    W   – win length
//    TS  – TT size (number of buckets, must be power of 2; default ≈ 4 M entries)
// ─────────────────────────────────────────────────────────────────────────────
template<int N, int W, int TS = (1 << 22)>
struct MinimaxRev4Agent {
    std::chrono::duration<double, std::milli> max_move_time;
    std::string name;

    MinimaxRev4Agent()
        : max_move_time(1.0), name("Unnamed MinimaxRev4Agent") { _init(); }
    MinimaxRev4Agent(double max_move_time_ms)
        : max_move_time(max_move_time_ms), name("Unnamed MinimaxRev4Agent") { _init(); }
    MinimaxRev4Agent(double max_move_time_ms, std::string n)
        : max_move_time(max_move_time_ms), name(std::move(n)) { _init(); }

    std::string& get_name() { return name; }

    int last_depth = -1;

    int get_last_depth() const { return last_depth; }

    // Returns a static evaluation for the current position.
    float get_eval(TicTacToe<N, W>& game) {
        tt.clear();
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

        // Clear all search state for a fresh move search.
        // TT is also cleared to prevent stale scores from earlier board positions
        // causing incorrect cutoffs (entries from 10+ moves ago share hash buckets).
        _clear_history();
        _clear_killers();
        tt.clear();

        int prev_move = -1;

        for (int depth = 0; ; ++depth) {
            int  cur_move  = -1;
            float cur_score = 0.0f;

            // Always use full window — aspiration windows waste time budget on retries
            // when the score changes between depths (common at deeper searches).
            {
                auto [m, s] = _root_search(game, -INF, INF, depth, deadline, seed);
                cur_move  = m;
                cur_score = s;
            }

            if (std::chrono::steady_clock::now() >= deadline) {
                if (prev_move == -1) {
                    // Couldn't finish even depth 0; emergency fallback.
                    for (int mv = 0; mv < N * N; ++mv)
                        if (game.at(mv) == BoardSquare::EMPTY) return mv;
                }
                return prev_move;
            }

            if (cur_move != -1) {
                prev_move = cur_move;
                last_depth = depth;
            }
            if (depth >= N * N) break;
        }
        return prev_move;
    }

private:
    // ── Constants ────────────────────────────────────────────────────────────
    static constexpr float INF              = 9999.0f;
    static constexpr float WIN_SCORE        = 1.0f;
    static constexpr float WIN_EPS          = 0.001f; // depth bonus per ply remaining
    static constexpr float ASPIRATION_DELTA = 0.15f;
    static constexpr float FUTILITY_MARGIN  = 0.5f;   // for depth-1 futility pruning
    static constexpr int   MAX_DEPTH        = N * N;
    static constexpr int   NUM_KILLERS      = 2;
    // Quiescence search depth cap (extra plies beyond depth=0 for tactical resolution).
    static constexpr int   QSEARCH_MAX_DEPTH   = 2;

    // ── Data ─────────────────────────────────────────────────────────────────
    FlatTT<TS> tt;

    // History table: history[player_idx][cell] — indexed by BoardSquare (X=0, O=1).
    std::array<std::array<int32_t, N * N>, 2> history{};

    // Killer moves: killers[ply][slot].
    std::array<std::array<int8_t, NUM_KILLERS>, MAX_DEPTH + 1> killers{};

    void _init() {
        _clear_history();
        _clear_killers();
    }

    void _clear_history() {
        for (auto& row : history) row.fill(0);
    }

    void _clear_killers() {
        for (auto& row : killers) row.fill(-1);
    }

    void _store_killer(int ply, int move) {
        if (ply < 0 || ply > MAX_DEPTH) return;
        auto& ks = killers[ply];
        if (ks[0] == move) return;         // already in slot 0
        ks[1] = ks[0];                     // shift
        ks[0] = static_cast<int8_t>(move);
    }

    bool _is_killer(int ply, int move) const {
        if (ply < 0 || ply > MAX_DEPTH) return false;
        const auto& ks = killers[ply];
        return ks[0] == move || ks[1] == move;
    }

    // ── Move ordering priority scores ────────────────────────────────────────
    // Returns a large float used to sort moves (highest first).
    // Priorities (descending):
    //   5000 : TT best move
    //   4000 : immediate win
    //   3000 : immediate block (opponent W-1 threat)
    //   2000 : fork (≥2 new threats)
    //   1500 : killer move
    //   1000 : single threat
    //    500 : center + history blend
    float _move_score(TicTacToe<N, W>& game, int move, int tt_best_move, int ply) const {
        if (move == tt_best_move) return 5000.0f;

        bool is_x = (game.next_player == BoardSquare::X);

        // Scan windows touching this cell.
        int win_moves      = 0; // windows where we complete W-in-a-row
        int block_moves    = 0; // windows where we block opponent W-1
        int new_threats    = 0; // windows where we create W-1 threat

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

        // Center proximity + history tiebreaker.
        float cx   = (N - 1) * 0.5f;
        float cy   = (N - 1) * 0.5f;
        float mx   = static_cast<float>(move % N);
        float my   = static_cast<float>(move / N);  // NOLINT: intentional integer division before cast
        float dist = std::abs(mx - cx) + std::abs(my - cy);
        float max_dist = cx + cy;
        float center = (max_dist > 0.0f) ? (1.0f - dist / max_dist) : 1.0f;

        int32_t hist = history[is_x ? 0 : 1][move];
        return 500.0f + center * 10.0f + std::log1p(static_cast<float>(hist));
    }

    // ── Static evaluation ─────────────────────────────────────────────────────
    // Uses the incremental threat-count differential for a sharper signal.
    // Return is from the perspective of the side to move (positive = good).
    float _leaf_eval(TicTacToe<N, W>& game, int prev_move) const {
        if (prev_move != -1) {
            BoardSquare w = game.check_winner(prev_move);
            // Winner is the side that just moved, which is the OPPONENT of next_player.
            if (w != BoardSquare::EMPTY) {
                // Opponent just won → bad for the side to move.
                return -(WIN_SCORE);
            }
        }
        // Blend static_eval (signed from X's perspective) with threat differential.
        float eval = std::clamp(game.static_eval, -1.0f, 1.0f);

        // W-1 threat bonus (immediate threats).
        constexpr float THREAT_WEIGHT = 0.05f;
        float threat_diff = static_cast<float>(game.x_threat_count - game.o_threat_count) * THREAT_WEIGHT;
        eval = std::clamp(eval + threat_diff, -1.0f, 1.0f);

        // W-2 threat bonus (potential open-ended threats), weighted less.
        if constexpr (W >= 3) {
            constexpr float THREAT2_WEIGHT = 0.015f;
            int x_threats2 = 0, o_threats2 = 0;
            for (const auto& ws : game.window_states) {
                if (ws.num_x == W - 2 && ws.num_o == 0) ++x_threats2;
                if (ws.num_o == W - 2 && ws.num_x == 0) ++o_threats2;
            }
            float threat2_diff = static_cast<float>(x_threats2 - o_threats2) * THREAT2_WEIGHT;
            eval = std::clamp(eval + threat2_diff, -1.0f, 1.0f);
        }

        // Convert to side-to-move perspective.
        return (game.next_player == BoardSquare::X) ? eval : -eval;
    }

    // ── Check for immediate forced win ───────────────────────────────────────
    // Returns the winning move index if the side to move can win in 1, else -1.
    int _find_immediate_win(TicTacToe<N, W>& game) const {
        bool is_x = (game.next_player == BoardSquare::X);
        for (int move = 0; move < N * N; ++move) {
            if (game.at(move) != BoardSquare::EMPTY) continue;
            for (int wi : game.cell_to_windows[move]) {
                const auto& ws = game.window_states[wi];
                if (is_x && ws.num_x == W - 1 && ws.num_o == 0) return move;
                if (!is_x && ws.num_o == W - 1 && ws.num_x == 0) return move;
            }
        }
        return -1;
    }

    // ── Quiescence search ─────────────────────────────────────────────────────
    // Called when depth reaches 0.  Keeps searching as long as either side has a
    // W-1 threat (a forcing move exists).  Caps at QSEARCH_MAX_DEPTH extra plies
    // to guarantee termination.  Uses a stand-pat score as a lower bound.
    float _qsearch(TicTacToe<N, W>& game, float alpha, float beta,
                   std::chrono::steady_clock::time_point deadline, int prev_move,
                   int ply, int qdepth = 0) {
        if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

        // Terminal: the side that just moved won.
        if (prev_move != -1) {
            BoardSquare w = game.check_winner(prev_move);
            if (w != BoardSquare::EMPTY) return -(WIN_SCORE);
        }

        // Stand-pat: evaluate the quiet position as a lower bound.
        float stand_pat = _leaf_eval(game, -1);
        if (stand_pat >= beta)  return stand_pat;
        if (stand_pat > alpha)  alpha = stand_pat;

        // If no tactical moves exist, or we've hit the qdepth cap, return stand-pat.
        bool any_threat = (game.x_threat_count > 0 || game.o_threat_count > 0);
        if (!any_threat || qdepth >= QSEARCH_MAX_DEPTH) return stand_pat;

        // Immediate win check — if we can win right now, do it.
        {
            int win_move = _find_immediate_win(game);
            if (win_move != -1) return WIN_SCORE;
        }

        // Only search tactical (threatening or blocking) moves.
        bool is_x = (game.next_player == BoardSquare::X);
        float best = stand_pat;

        for (int mv = 0; mv < N * N; ++mv) {
            if (game.at(mv) != BoardSquare::EMPTY) continue;

            // Check if this move is tactical: wins, blocks, or creates a W-1 threat.
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

    // ── Negamax (PVS + alpha-beta + all pruning) ─────────────────────────────
    // Returns score from the perspective of the side to move (positive = good for mover).
    // `ply` is distance from root (0 at root), used for killer indexing.
    float _negamax(TicTacToe<N, W>& game, float alpha, float beta, int depth,
                   std::chrono::steady_clock::time_point deadline, int prev_move, int ply) {
        if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

        // ── Terminal: winner check ────────────────────────────────────────────
        if (prev_move != -1) {
            BoardSquare w = game.check_winner(prev_move);
            if (w != BoardSquare::EMPTY) {
                // The side that just moved won.  That is the opponent of next_player.
                // Score is negative for the side to move (they just lost).
                return -(WIN_SCORE + depth * WIN_EPS);
            }
        }

        // ── Leaf node: enter quiescence search instead of raw eval ─────────────
        if (depth <= 0) return _qsearch(game, alpha, beta, deadline, prev_move, ply);

        // ── TT probe ─────────────────────────────────────────────────────────
        long long key      = game.hash_val;
        int       tt_best  = -1;
        {
            const auto* e = tt.probe(key);
            if (e) {
                tt_best = e->best_move;
                if (e->depth >= depth) {
                    float s = e->score;
                    if (e->type == NodeType::EXACT) return s;
                    if (e->type == NodeType::LOWER) alpha = std::max(alpha, s);
                    else                            beta  = std::min(beta,  s);
                    if (alpha >= beta) return s;
                }
            }
        }

        // ── Immediate-win pruning ─────────────────────────────────────────────
        // If side to move can win right now, no need to explore anything else.
        {
            int win_move = _find_immediate_win(game);
            if (win_move != -1) {
                float score = WIN_SCORE + depth * WIN_EPS;
                tt.store(key, score, depth, NodeType::EXACT, win_move);
                return score;
            }
        }

        // ── Futility pruning (depth == 1) ─────────────────────────────────────
        // If our static eval + a generous margin still can't beat alpha, skip non-tactical moves.
        bool futility_active = false;
        float futility_base  = 0.0f;
        if (depth == 1) {
            futility_base   = _leaf_eval(game, -1); // no prev_move: pure positional
            futility_active = (futility_base + FUTILITY_MARGIN < alpha);
        }

        // ── Generate and sort moves ───────────────────────────────────────────
        std::array<std::pair<float, int>, N * N> move_buf;
        int count = 0;
        for (int mv = 0; mv < N * N; ++mv) {
            if (game.at(mv) != BoardSquare::EMPTY) continue;
            float ms = _move_score(game, mv, tt_best, ply);
            move_buf[count++] = {ms, mv};
        }

        if (count == 0) return 0.0f; // board full → draw

        std::sort(move_buf.begin(), move_buf.begin() + count,
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // ── Alpha-beta with PVS ───────────────────────────────────────────────
        const float alpha_orig = alpha;
        const float beta_orig  = beta;
        float best_score = -INF;
        int   best_move  = move_buf[0].second;

        for (int i = 0; i < count; ++i) {
            if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

            int mv = move_buf[i].second;

            // Futility pruning: skip quiet (non-tactical) moves at depth 1.
            if (futility_active && move_buf[i].first < 1000.0f) {
                // move_score < 1000 means no immediate win / block / fork / killer
                continue;
            }

            game.play_move(mv);

            float val;
            if (i == 0) {
                // Full-window search for the first (best-ordered) child.
                val = -_negamax(game, -beta, -alpha, depth - 1, deadline, mv, ply + 1);
            } else {
                // Null-window search (PVS) at full depth for all non-first moves.
                // LMR is intentionally not applied: this game is highly tactical and
                // LMR incorrectly dismisses moves that are critical at deeper searches.
                val = -_negamax(game, -alpha - 1e-4f, -alpha, depth - 1, deadline, mv, ply + 1);

                // Re-search at full window if it failed high.
                if (val > alpha && std::chrono::steady_clock::now() < deadline) {
                    val = -_negamax(game, -beta, -alpha, depth - 1, deadline, mv, ply + 1);
                }
            }

            game.unplay_move(mv);

            // Discard result if deadline was hit during the recursive call — the
            // returned 0.0f would corrupt alpha and mislead subsequent siblings.
            if (std::chrono::steady_clock::now() >= deadline) return 0.0f;

            if (val > best_score) {
                best_score = val;
                best_move  = mv;
            }
            alpha = std::max(alpha, val);
            if (alpha >= beta) {
                // Beta cutoff — store killer and update history.
                _store_killer(ply, mv);
                history[game.next_player == BoardSquare::X ? 0 : 1][mv] += depth * depth;
                break;
            }
        }

        // ── TT store ─────────────────────────────────────────────────────────
        if (std::chrono::steady_clock::now() < deadline) {
            NodeType type = NodeType::EXACT;
            if (best_score <= alpha_orig) type = NodeType::UPPER;
            else if (best_score >= beta_orig)  type = NodeType::LOWER;
            tt.store(key, best_score, depth, type, best_move);
        }

        return best_score;
    }

    // ── Root search for one IDDFS depth ──────────────────────────────────────
    // Returns {best_move, best_score} in the mover's frame.
    // alpha/beta passed in so the caller can implement aspiration windows.
    std::pair<int, float> _root_search(TicTacToe<N, W>& game,
                                       float alpha, float beta,
                                       int depth,
                                       std::chrono::steady_clock::time_point deadline,
                                       int seed) {
        // Probe TT for a best move hint from a previous IDDFS iteration.
        int tt_best = -1;
        {
            const auto* e = tt.probe(game.hash_val);
            if (e) tt_best = e->best_move;
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

        // Quick check: if depth == 0 just pick best-ordered move.
        if (depth == 0) {
            return {move_buf[0].second, _move_score(game, move_buf[0].second, tt_best, 0)};
        }

        float best_score = -INF;
        int   best_move  = move_buf[0].second;

        // Collect all moves tied for best so we can break ties randomly.
        std::vector<int> best_moves;

        const float alpha_orig = alpha;
        const float beta_orig  = beta;

        for (int i = 0; i < count; ++i) {
            if (std::chrono::steady_clock::now() >= deadline) break;

            int mv = move_buf[i].second;

            game.play_move(mv);
            float val;
            if (i == 0) {
                val = -_negamax(game, -beta, -alpha, depth - 1, deadline, mv, 1);
            } else {
                val = -_negamax(game, -alpha - 1e-4f, -alpha, depth - 1, deadline, mv, 1);
                if (val > alpha && val < beta && std::chrono::steady_clock::now() < deadline)
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

        // Break ties randomly using the caller's seed.
        if (best_moves.size() > 1) {
            std::mt19937 gen(static_cast<unsigned>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
            std::uniform_int_distribution<int> dist(0, static_cast<int>(best_moves.size()) - 1);
            best_move = best_moves[dist(gen)];
        }

        // Store root result in TT.
        if (std::chrono::steady_clock::now() < deadline) {
            NodeType type = NodeType::EXACT;
            if (best_score <= alpha_orig) type = NodeType::UPPER;
            else if (best_score >= beta_orig) type = NodeType::LOWER;
            tt.store(game.hash_val, best_score, depth, type, best_move);
        }

        return {best_move, best_score};
    }
};
