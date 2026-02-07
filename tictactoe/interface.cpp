// g++ -std=c++23 -O3 -o play.exe tictactoe/interface.cpp && play.exe

#include "tictactoe.hpp"
#include "../FFNN.hpp"

void play() {
    TicTacToe game;
    FFNN ffnn = FFNN::from_file("tictactoe/network.dat");

    for (int i = 0; i < 9; i++) {
        if (game.next_player == BoardSquare::X) {
            int player_move_x;
            int player_move_y;
            std::string player_move;

            game.print_board();
            std::cout << "Player '" << square_to_char(game.next_player) << "' (xy): ";

            if (!(std::cin >> player_move)) {
                std::cin.clear();
                continue;
            }
            std::cout << std::endl;

            if (isdigit(player_move[0])) player_move_x = player_move[0] - '0';
            else continue;
            if (isdigit(player_move[1])) player_move_y = player_move[1] - '0';
            else continue;

            if (player_move_x < 0 || player_move_y < 0) continue;
            if (player_move_x > 2 || player_move_y > 2) continue;
        
            if (!game.play_move(player_move_x, player_move_y)) std::cout << "Could Not Play Move, Try again" << std::endl;

            
        } else {
            Eigen::MatrixXf board_state = game.get_board_state(game.next_player);
            Eigen::MatrixXf move_probabilities = ffnn.forward(board_state);
            int move_index;
            move_probabilities.col(0).maxCoeff(&move_index);
            bool move_succeeded = game.play_move(move_index);

            if (!move_succeeded) {
                // that move was invalid, sort the matrix and play moves in order of likelyhood untill one works
                std::vector<std::pair<float, int>> probabilities(9);
                for (int i = 0; i < 9; i++) {
                    probabilities[i] = std::pair<float, int>(move_probabilities(i), i);
                }

                std::sort(probabilities.begin(), probabilities.end(), [](const auto& a, const auto&  b){ return a.first > b.first; });

                for (int i = 1; i < 9; i++) {
                    if (game.play_move(probabilities[i].second)) {
                        move_index = probabilities[i].second;
                        break;
                    }
                }
            }
        }
        
        if (game.check_winner() != BoardSquare::EMPTY) {
            game.print_board();
            std::cout << square_to_char(game.check_winner()) << " is the winner!" << std::endl;
            break;
        }
    }
}

int main() {
    play();
    return 0;
}