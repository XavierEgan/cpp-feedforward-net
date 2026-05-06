#pragma once
#include "./enums.hpp"

#include "../Eigen/Dense"
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <unordered_map>

// NxN board, W win length
template<int N, int W>
struct TicTacToe {
    static_assert(N >= 2,          "TicTacToe requires N >= 2");
    static_assert(W >= 2 && W <= N, "TicTacToe requires 2 <= W <= N");

    BoardSquare next_player;
    std::array<BoardSquare, N * N> board;

    // Per-window piece counts maintained incrementally during play/unplay.
    struct WindowState {
        int8_t num_x = 0;
        int8_t num_o = 0;
    };

    // Indices of the W cells that form a window.
    struct WindowDef {
        std::array<int, W> cells;
    };

    // Fixed geometry, computed once in constructor.
    std::vector<WindowDef>              window_defs;
    std::array<std::vector<int>, N * N> cell_to_windows;

    // Incremental state – all reset by restart().
    std::vector<WindowState> window_states;
    float static_eval    = 0.0f; // running heuristic sum; clamp to [-1,1] before use
    int   x_threat_count = 0;    // windows with num_x == W-1 and num_o == 0
    int   o_threat_count = 0;    // windows with num_o == W-1 and num_x == 0

    // Incremental Zobrist hash.
    std::array<long long, N * N * 2> zobrist_keys; // [cell*2+0]=X piece, [cell*2+1]=O piece
    long long hash_x_to_move = 0;
    long long hash_val       = 0;

    TicTacToe() {
        board.fill(BoardSquare::EMPTY);
        next_player = BoardSquare::X;
        _build_windows();
        _init_zobrist();
    }

    void restart() {
        board.fill(BoardSquare::EMPTY);
        next_player = BoardSquare::X;
        _reset_incremental();
    }

    BoardSquare at(int index) const { return board[index]; }
    BoardSquare& at(int index)      { return board[index]; }
    BoardSquare at(int x, int y) const { return board[y * N + x]; }
    BoardSquare& at(int x, int y)      { return board[y * N + x]; }

    /*
    return a matrix representation of the board with
    1 = X, -1 = O, 0 = blank
    */
    Eigen::MatrixXf get_board_state() {
        Eigen::MatrixXf state(N * N, 1);
        for (int i = 0; i < N * N; i++) {
            if      (board[i] == BoardSquare::X) state(i, 0) =  1.0f;
            else if (board[i] == BoardSquare::O) state(i, 0) = -1.0f;
            else                                 state(i, 0) =  0.0f;
        }
        return state;
    }

    void print_board() {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++)
                std::cout << ' ' << square_to_char(at(x, y)) << ' ' << (x != N - 1 ? '|' : '\n');
            if (y != N - 1) std::cout << std::string(N * 4 - 1, '-') << std::endl;
        }
    }

    // Returns the running count of W-1-in-a-row threats for the given player.
    int get_threats(BoardSquare player) const {
        return (player == BoardSquare::X) ? x_threat_count : o_threat_count;
    }

    bool play_move(int i) {
        if (at(i) != BoardSquare::EMPTY) return false;
        const bool is_x = (next_player == BoardSquare::X);

        for (int wi : cell_to_windows[i]) {
            WindowState& ws = window_states[wi];
            static_eval -= _window_score(ws);
            _adjust_threats(ws, -1);
            if (is_x) ws.num_x++; else ws.num_o++;
            static_eval += _window_score(ws);
            _adjust_threats(ws, +1);
        }

        hash_val   ^= zobrist_keys[i * 2 + (is_x ? 0 : 1)];
        at(i)       = next_player;
        next_player = (next_player == BoardSquare::O) ? BoardSquare::X : BoardSquare::O;
        hash_val   ^= hash_x_to_move;
        return true;
    }

    bool unplay_move(int i) {
        if (at(i) == BoardSquare::EMPTY) return false;
        const bool was_x = (at(i) == BoardSquare::X);

        for (int wi : cell_to_windows[i]) {
            WindowState& ws = window_states[wi];
            static_eval -= _window_score(ws);
            _adjust_threats(ws, -1);
            if (was_x) ws.num_x--; else ws.num_o--;
            static_eval += _window_score(ws);
            _adjust_threats(ws, +1);
        }

        hash_val   ^= zobrist_keys[i * 2 + (was_x ? 0 : 1)];
        at(i)       = BoardSquare::EMPTY;
        next_player = (next_player == BoardSquare::O) ? BoardSquare::X : BoardSquare::O;
        hash_val   ^= hash_x_to_move;
        return true;
    }

    // returns BoardSquare::EMPTY if there is no winner
    // returns the winner if there is a winner
    BoardSquare check_winner(int most_recent_move) {
        int x = most_recent_move % N;
        int y = most_recent_move / N;

        BoardSquare p = at(most_recent_move);
        if (p == BoardSquare::EMPTY) return BoardSquare::EMPTY;

        auto count_dir = [&](int dx, int dy) {
            int c = 0;
            int cx = x + dx, cy = y + dy;
            while (cx >= 0 && cx < N && cy >= 0 && cy < N && at(cx, cy) == p) {
                ++c;
                cx += dx;
                cy += dy;
            }
            return c;
        };

        // horizontal
        if (1 + count_dir(-1, 0) + count_dir(1, 0) >= W) return p;
        // vertical
        if (1 + count_dir(0, -1) + count_dir(0, 1) >= W) return p;
        // diagonal TL-BR
        if (1 + count_dir(-1, -1) + count_dir(1, 1) >= W) return p;
        // diagonal TR-BL
        if (1 + count_dir(1, -1) + count_dir(-1, 1) >= W) return p;

        return BoardSquare::EMPTY;
    }

private:
    // Build the sliding-window geometry for all four directions.
    void _build_windows() {
        auto add_window = [&](int sx, int sy, int dx, int dy) {
            const int idx = (int)window_defs.size();
            WindowDef wd;
            for (int k = 0; k < W; k++) {
                wd.cells[k] = (sy + k * dy) * N + (sx + k * dx);
                cell_to_windows[wd.cells[k]].push_back(idx);
            }
            window_defs.push_back(wd);
        };

        for (int y = 0; y < N; y++)          // rows
            for (int x = 0; x <= N - W; x++)
                add_window(x, y,  1,  0);

        for (int x = 0; x < N; x++)          // cols
            for (int y = 0; y <= N - W; y++)
                add_window(x, y,  0,  1);

        for (int y = 0; y <= N - W; y++)     // diag TL->BR
            for (int x = 0; x <= N - W; x++)
                add_window(x, y,  1,  1);

        for (int y = 0; y <= N - W; y++)     // diag TR->BL
            for (int x = W - 1; x < N; x++)
                add_window(x, y, -1,  1);

        window_states.resize(window_defs.size());
    }

    void _init_zobrist() {
        std::mt19937_64 rng(std::random_device{}());
        for (auto& k : zobrist_keys) k = static_cast<long long>(rng());
        hash_x_to_move = static_cast<long long>(rng());
        hash_val = hash_x_to_move; // X moves first
    }

    void _reset_incremental() {
        for (auto& ws : window_states) ws = WindowState{};
        static_eval    = 0.0f;
        x_threat_count = 0;
        o_threat_count = 0;
        hash_val       = hash_x_to_move; // X moves first
    }

    // Heuristic contribution of a single window to static_eval.
    // Matches the per-window scoring used in MinimaxRev3's original eval.
    static float _window_score(const WindowState& ws) {
        constexpr float th = 0.1f;
        if (ws.num_x > 0 && ws.num_o > 0) return 0.0f;
        if (ws.num_o == 0 && ws.num_x > 0) {
            if (ws.num_x == W - 1) return th;
            return th * (static_cast<float>(ws.num_x) / W);
        }
        if (ws.num_x == 0 && ws.num_o > 0) {
            if (ws.num_o == W - 1) return -th;
            return -th * (static_cast<float>(ws.num_o) / W);
        }
        return 0.0f;
    }

    void _adjust_threats(const WindowState& ws, int delta) {
        if (ws.num_x == W - 1 && ws.num_o == 0) x_threat_count += delta;
        if (ws.num_o == W - 1 && ws.num_x == 0) o_threat_count += delta;
    }
};