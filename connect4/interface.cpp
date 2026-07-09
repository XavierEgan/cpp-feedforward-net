// g++ -std=c++23 -O3 -o c4_interface connect4/interface.cpp && ./c4_interface
//
// usage (run from the repo root):
//   ./build/c4_interface                benchmark: negamax vs random, then strong vs weak negamax
//   ./build/c4_interface human          play against the negamax agent in the terminal
//   ./build/c4_interface ffnn           play against the trained net (needs connect4/models/best.dat)

#include "./connect4.hpp"
#include "./Agent.hpp"
#include "./agentTools.hpp"
#include "../FFNN.hpp"

#include <iostream>
#include <string>

constexpr int width = 7;
constexpr int height = 6;

void run_benchmarks() {
    C4NegamaxAgent<width, height> strong(30.0, "negamax-30ms");
    C4NegamaxAgent<width, height> weak(1.0, "negamax-1ms");
    C4RandomAgent<width, height> random("random");

    std::cout << "=== negamax-1ms vs random ===" << std::endl;
    c4_benchmark_agents<width, height>(weak, random, 100, true);

    std::cout << "\n=== negamax-30ms vs negamax-1ms ===" << std::endl;
    c4_benchmark_agents<width, height>(strong, weak, 30, true);
}

template<typename Opponent>
void play_human(Opponent& opponent) {
    Connect4<width, height> game;
    C4HumanAgent<width, height> human("human");

    while (!game.is_full()) {
        int move;
        if (game.next_player() == Disc::RED) move = human.get_move(game);
        else move = opponent.get_move(game);
        game.play_move(move);

        const Disc winner = game.check_winner();
        if (winner != Disc::EMPTY) {
            game.print_board();
            std::cout << disc_to_char(winner) << " wins!" << std::endl;
            return;
        }
    }

    game.print_board();
    std::cout << "draw" << std::endl;
}

int main(int argc, const char* argv[]) {
    const std::string mode = (argc > 1) ? argv[1] : "";

    if (mode == "human") {
        C4NegamaxAgent<width, height> opponent(100.0, "negamax-100ms");
        play_human(opponent);
    } else if (mode == "ffnn") {
        FFNN model = FFNN::from_file("connect4/models/best.dat");
        C4FFNNAgent<width, height> opponent(50.0, "ffnn-50ms", model);
        play_human(opponent);
    } else {
        run_benchmarks();
    }

    return 0;
}
