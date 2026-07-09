#pragma once
#include "./enums.hpp"

#include "../Eigen/Dense"

#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <string>

/*
bitboard connect 4 engine (fixed win length of 4)

each column is HEIGHT+1 bits tall; the extra sentinel bit above each column stops
vertical/diagonal shift tricks bleeding into the next column. bit index for cell
(x, y) is x * (HEIGHT + 1) + y, with y = 0 at the BOTTOM of the board.

state is two bitboards:
    pos  – discs belonging to the side to move
    mask – all discs

play is three bit ops, unplay pops a move history stack, and pos + mask is a
perfect hash of the position (no zobrist needed). win detection is four
shift-and-ands. this is the classic Fhourstones representation and is orders of
magnitude faster than the array board used by tictactoe/
*/
template<int WIDTH = 7, int HEIGHT = 6>
struct Connect4 {
    static_assert(WIDTH >= 4 && HEIGHT >= 4, "Connect4 requires WIDTH >= 4 and HEIGHT >= 4");
    static_assert(WIDTH * (HEIGHT + 1) <= 64, "Connect4 board must fit in a 64-bit bitboard");

    static constexpr int num_cells = WIDTH * HEIGHT;
    static constexpr int col_bits = HEIGHT + 1; // bits per column including sentinel

    uint64_t pos = 0;  // discs of the side to move
    uint64_t mask = 0; // all discs
    int num_moves = 0;
    std::array<int8_t, num_cells> history{}; // columns played, for unplay_move

    // ── static geometry ──────────────────────────────────────────────────────

    static constexpr uint64_t bottom_mask(int col) {
        return uint64_t(1) << (col * col_bits);
    }

    static constexpr uint64_t top_mask(int col) {
        return uint64_t(1) << (HEIGHT - 1 + col * col_bits);
    }

    static constexpr uint64_t column_mask(int col) {
        return ((uint64_t(1) << HEIGHT) - 1) << (col * col_bits);
    }

    // playable cells across the whole board (sentinel row excluded)
    static constexpr uint64_t board_mask() {
        uint64_t m = 0;
        for (int col = 0; col < WIDTH; col++) m |= column_mask(col);
        return m;
    }

    // one bit per column in the bottom row
    static constexpr uint64_t bottom_row_mask() {
        uint64_t m = 0;
        for (int col = 0; col < WIDTH; col++) m |= bottom_mask(col);
        return m;
    }

    // ── game state ───────────────────────────────────────────────────────────

    void restart() {
        pos = 0;
        mask = 0;
        num_moves = 0;
    }

    Disc next_player() const {
        return (num_moves % 2 == 0) ? Disc::RED : Disc::YELLOW;
    }

    // bitboard of one player's discs
    uint64_t discs(Disc player) const {
        return (player == next_player()) ? pos : (pos ^ mask);
    }

    bool can_play(int col) const {
        if (col < 0 || col >= WIDTH) return false;
        return (mask & top_mask(col)) == 0;
    }

    bool play_move(int col) {
        if (!can_play(col)) return false;
        pos ^= mask;                          // switch perspective to the opponent
        mask |= mask + bottom_mask(col);      // drop a disc into col
        history[num_moves++] = static_cast<int8_t>(col);
        return true;
    }

    bool unplay_move() {
        if (num_moves == 0) return false;
        const int col = history[--num_moves];
        const uint64_t col_discs = mask & column_mask(col);
        const uint64_t top_disc = uint64_t(1) << (63 - std::countl_zero(col_discs));
        mask ^= top_disc;                     // remove the highest disc in col
        pos ^= mask;                          // switch perspective back
        return true;
    }

    // perfect hash of the position (pos + mask is injective for valid states)
    uint64_t key() const {
        return pos + mask;
    }

    bool is_full() const {
        return num_moves == num_cells;
    }

    // one bit per column at the cell the next disc would land in
    uint64_t possible_moves() const {
        return (mask + bottom_row_mask()) & board_mask();
    }

    // ── win detection ────────────────────────────────────────────────────────

    // true if the given bitboard contains 4 in a row in any direction
    static bool has_won(uint64_t b) {
        // vertical
        uint64_t m = b & (b >> 1);
        if (m & (m >> 2)) return true;
        // horizontal
        m = b & (b >> col_bits);
        if (m & (m >> (2 * col_bits))) return true;
        // diagonal \ (next column, one row down)
        m = b & (b >> HEIGHT);
        if (m & (m >> (2 * HEIGHT))) return true;
        // diagonal / (next column, one row up)
        m = b & (b >> (HEIGHT + 2));
        if (m & (m >> (2 * (HEIGHT + 2)))) return true;
        return false;
    }

    // returns Disc::EMPTY if there is no winner, otherwise the winner
    // only the last mover can have won, so this is a single has_won call
    Disc check_winner() const {
        if (num_moves == 0) return Disc::EMPTY;
        if (!has_won(pos ^ mask)) return Disc::EMPTY;
        return (num_moves % 2 == 1) ? Disc::RED : Disc::YELLOW;
    }

    /*
    empty cells that would complete 4 in a row for `position` (not necessarily
    playable yet — a disc may need to land below them first). used for the
    static eval and for immediate win/block detection in search
    */
    static uint64_t winning_spots(uint64_t position, uint64_t all_discs) {
        // vertical
        uint64_t r = (position << 1) & (position << 2) & (position << 3);

        // horizontal, diagonal \, diagonal / share the same pattern with
        // different shift amounts
        for (int shift : {col_bits, HEIGHT, HEIGHT + 2}) {
            // pairs to the left of the candidate cell
            uint64_t p = (position << shift) & (position << (2 * shift));
            r |= p & (position << (3 * shift)); // _XXX
            r |= p & (position >> shift);       // X_XX
            // pairs to the right of the candidate cell
            p = (position >> shift) & (position >> (2 * shift));
            r |= p & (position >> (3 * shift)); // XXX_
            r |= p & (position << shift);       // XX_X
        }

        return r & (board_mask() ^ all_discs);
    }

    // ── board access and display ─────────────────────────────────────────────

    Disc at(int x, int y) const {
        const uint64_t bit = uint64_t(1) << (x * col_bits + y);
        if ((mask & bit) == 0) return Disc::EMPTY;
        return (pos & bit) ? next_player() : other_player(next_player());
    }

    /*
    return a matrix representation of the board with
    1 = RED, -1 = YELLOW, 0 = blank
    index = y * WIDTH + x, y = 0 at the bottom
    */
    Eigen::MatrixXf get_board_state() const {
        Eigen::MatrixXf state(num_cells, 1);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                const Disc d = at(x, y);
                float v = 0.0f;
                if (d == Disc::RED) v = 1.0f;
                else if (d == Disc::YELLOW) v = -1.0f;
                state(y * WIDTH + x, 0) = v;
            }
        }
        return state;
    }

    void print_board() const {
        for (int y = HEIGHT - 1; y >= 0; y--) {
            for (int x = 0; x < WIDTH; x++)
                std::cout << '|' << ' ' << disc_to_char(at(x, y)) << ' ';
            std::cout << '|' << '\n';
        }
        std::cout << std::string(WIDTH * 4 + 1, '-') << '\n';
        for (int x = 0; x < WIDTH; x++)
            std::cout << ' ' << ' ' << x << ' ';
        std::cout << std::endl;
    }
};
