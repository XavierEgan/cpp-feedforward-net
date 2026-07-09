// clang++ -std=c++23 -O0 -I.. tests/test_connect4.cpp -o test_connect4 && ./test_connect4

#include "../connect4/connect4.hpp"
#include "../connect4/enums.hpp"

#include <bit>
#include <cassert>
#include <iostream>
#include <random>

using Game = Connect4<7, 6>;

Game play_sequence(std::initializer_list<int> cols) {
    Game game;
    for (int col : cols) assert(game.play_move(col));
    return game;
}

void test_vertical_win() {
    // red stacks column 3, yellow follows in column 0
    Game game = play_sequence({3, 0, 3, 0, 3, 0, 3});
    assert(game.check_winner() == Disc::RED);
    std::cout << "vertical win: PASS\n";
}

void test_horizontal_win() {
    // red plays 0..3 across the bottom, yellow stacks column 6
    Game game = play_sequence({0, 6, 1, 6, 2, 6, 3});
    assert(game.check_winner() == Disc::RED);
    std::cout << "horizontal win: PASS\n";
}

void test_diagonal_win() {
    // classic staircase: red on the / diagonal from (0,0) to (3,3)
    Game game = play_sequence({0, 1, 1, 2, 2, 3, 2, 3, 3, 0, 3});
    assert(game.check_winner() == Disc::RED);
    std::cout << "diagonal win: PASS\n";
}

void test_yellow_win() {
    Game game = play_sequence({0, 3, 0, 3, 1, 3, 1, 3});
    assert(game.check_winner() == Disc::YELLOW);
    std::cout << "yellow win: PASS\n";
}

void test_no_winner_midgame() {
    Game game = play_sequence({3, 3, 2, 4});
    assert(game.check_winner() == Disc::EMPTY);
    assert(game.next_player() == Disc::RED);
    std::cout << "no winner midgame: PASS\n";
}

void test_column_fills_up() {
    Game game;
    for (int i = 0; i < 6; i++) assert(game.play_move(0));
    assert(!game.can_play(0));
    assert(!game.play_move(0));
    assert(game.can_play(1));
    std::cout << "column fills up: PASS\n";
}

void test_play_unplay_roundtrip() {
    std::mt19937 rng(42);
    Game game;

    for (int trial = 0; trial < 200; trial++) {
        game.restart();

        // random playout, snapshotting and undoing each move
        while (!game.is_full() && game.check_winner() == Disc::EMPTY) {
            std::vector<int> legal;
            for (int col = 0; col < 7; col++)
                if (game.can_play(col)) legal.push_back(col);
            std::uniform_int_distribution<int> dist(0, static_cast<int>(legal.size()) - 1);
            const int col = legal.at(dist(rng));

            const uint64_t pos_before = game.pos;
            const uint64_t mask_before = game.mask;
            const int moves_before = game.num_moves;

            assert(game.play_move(col));
            assert(game.num_moves == moves_before + 1);
            assert(game.unplay_move());

            assert(game.pos == pos_before);
            assert(game.mask == mask_before);
            assert(game.num_moves == moves_before);

            game.play_move(col); // replay and continue
        }

        // unwind the whole game back to empty
        while (game.num_moves > 0) assert(game.unplay_move());
        assert(game.pos == 0 && game.mask == 0);
    }
    std::cout << "play/unplay roundtrip: PASS\n";
}

void test_board_state_matches_at() {
    Game game = play_sequence({3, 3, 2, 4, 0});
    const Eigen::MatrixXf state = game.get_board_state();

    for (int y = 0; y < 6; y++) {
        for (int x = 0; x < 7; x++) {
            const Disc d = game.at(x, y);
            const float v = state(y * 7 + x, 0);
            if (d == Disc::RED) assert(v == 1.0f);
            else if (d == Disc::YELLOW) assert(v == -1.0f);
            else assert(v == 0.0f);
        }
    }
    // reds at (3,0), (2,0), (0,0); yellows at (3,1), (4,0)
    assert(game.at(3, 0) == Disc::RED);
    assert(game.at(3, 1) == Disc::YELLOW);
    assert(game.at(4, 0) == Disc::YELLOW);
    assert(game.at(0, 0) == Disc::RED);
    std::cout << "board state matches at(): PASS\n";
}

void test_winning_spots() {
    // red has (1,0), (2,0), (3,0): both (0,0) and (4,0) complete the row
    Game game = play_sequence({1, 1, 2, 2, 3, 3});
    const uint64_t red = game.discs(Disc::RED);
    const uint64_t spots = Game::winning_spots(red, game.mask);
    assert(std::popcount(spots) == 2);
    assert(spots & Game::bottom_mask(0));
    assert(spots & Game::bottom_mask(4));
    std::cout << "winning spots: PASS\n";
}

void test_key_changes_with_position() {
    Game a = play_sequence({3});
    Game b = play_sequence({4});
    Game c = play_sequence({3, 4});
    assert(a.key() != b.key());
    assert(a.key() != c.key());
    // same position reached in a different move order has the same key
    Game d = play_sequence({3, 4, 2, 5});
    Game e = play_sequence({2, 5, 3, 4});
    assert(d.key() == e.key());
    std::cout << "position key: PASS\n";
}

int main() {
    test_vertical_win();
    test_horizontal_win();
    test_diagonal_win();
    test_yellow_win();
    test_no_winner_midgame();
    test_column_fills_up();
    test_play_unplay_roundtrip();
    test_board_state_matches_at();
    test_winning_spots();
    test_key_changes_with_position();

    std::cout << "ALL PASS\n";
    return 0;
}
