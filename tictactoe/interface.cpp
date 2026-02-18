// g++ -std=c++23 -O3 -o play.exe tictactoe/interface.cpp && play.exe

#include "tictactoe.hpp"
#include "../FFNN.hpp"

void play() {
    const int board_size = 4;
    TicTacToe<board_size> game;
    FFNN ffnn = FFNN::from_file("tictactoe/perfect_4x4.dat");

    for (int i = 0; i < board_size * board_size; i++) {
        if (game.next_player == BoardSquare::O) {
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
            if (player_move_x >= board_size || player_move_y >= board_size) continue;
        
            if (!game.play_move(player_move_y * board_size + player_move_x)) std::cout << "Could Not Play Move, Try again" << std::endl;

            
        } else {
            Eigen::MatrixXf board_state = game.get_board_state(game.next_player);
            Eigen::MatrixXf move_probabilities = ffnn.forward(board_state);
            
            game.play_move(move_probabilities);
            
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