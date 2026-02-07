// g++ -std=c++23 -O3 -o train.exe tictactoe/train.cpp && train.exe

#include "tictactoe.hpp"
#include "../AdamOptimiser.hpp"
#include "../NN_Utils.hpp"
#include <vector>
#include <algorithm>
#include <iostream>

/*
play against yourself for a bit and if you win then take all the moves in the game as positive examples and vice versa
train on that and repeat
*/
void naive() {
    std::vector<size_t> network_shape {9, 16, 16, 9};
    std::vector<ActivationFunc> activation_funcs {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax};

    FFNN ffnn = FFNN::from_random_he_scaling(network_shape, activation_funcs);
    //FFNN ffnn = FFNN::from_file("tictactoe/network.dat");

    int num_epochs = 2000;
    int minibatch_size = -1;
    int num_batch_training = 10;
    int num_games = 1000;
    double epsilon = 0.2;
    double epsilon_decay = 0.95;

    TicTacToe tic_tac_toe;
    AdamOptimiser optimiser(ffnn, CostType::categorical_cross_entropy);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // get training data
        std::vector<Eigen::MatrixXf> inputs;
        std::vector<Eigen::MatrixXf> targets;

        BoardSquare winner;

        // board state before the player takes a move
        std::vector<Eigen::MatrixXf> player_x_game_history;
        std::vector<Eigen::MatrixXf> player_o_game_history;
        std::vector<int> player_x_moves;
        std::vector<int> player_o_moves;

        // logging/debugging
        int num_draws = 0;

        for (int game = 0; game < num_games; game++) {
            tic_tac_toe.restart();
            
            player_x_game_history.clear();
            player_o_game_history.clear();
            player_x_moves.clear();
            player_o_moves.clear();

            // if we do 9 iterations and nobody one then the game is a draw
            for (int i = 0; i < 9; i++) {
                // get board state
                Eigen::MatrixXf board_state = tic_tac_toe.get_board_state(tic_tac_toe.next_player);

                // note down the board state
                if (tic_tac_toe.next_player == BoardSquare::O) player_o_game_history.push_back(board_state);
                else if (tic_tac_toe.next_player == BoardSquare::X) player_x_game_history.push_back(board_state);

                // get network prediction
                Eigen::MatrixXf move_probabilities = ffnn.forward(board_state);
                int move_index;
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> uniform(0.0, 1.0);

                if (uniform(gen) < epsilon) {
                    std::uniform_int_distribution<> random_move(0, 8);
                    move_index = random_move(gen);
                } else {
                    std::discrete_distribution<> distrib(move_probabilities.data(), move_probabilities.data() + move_probabilities.size());
                    move_index = distrib(gen);
                }
                

                // try do the move
                bool move_succeeded = tic_tac_toe.play_move(move_index);
                
                if (!move_succeeded) {
                    // that move was invalid, sort the matrix and play moves in order of likelyhood untill one works
                    std::vector<std::pair<float, int>> probabilities(9);
                    for (int i = 0; i < 9; i++) {
                        probabilities[i] = std::pair<float, int>(move_probabilities(i), i);
                    }

                    std::sort(probabilities.begin(), probabilities.end(), [](const auto& a, const auto&  b){ return a.first > b.first; });

                    for (int i = 1; i < 9; i++) {
                        if (tic_tac_toe.play_move(probabilities[i].second)) {
                            move_index = probabilities[i].second;
                            break;
                        }
                    }
                }

                // note down the move played (opposite because the next player changes when we play the move)
                if (tic_tac_toe.next_player == BoardSquare::O) player_x_moves.push_back(move_index);
                else if (tic_tac_toe.next_player == BoardSquare::X) player_o_moves.push_back(move_index);
                
                winner = tic_tac_toe.check_winner();
                if (winner != BoardSquare::EMPTY) {
                    break;
                };
            }

            if (winner == BoardSquare::X) {
                for (const auto& b : player_x_game_history) inputs.push_back(b);
                for (int move : player_x_moves) {
                    Eigen::MatrixXf probability_matrix = Eigen::MatrixXf::Zero(9, 1);
                    probability_matrix(move) = 1.0f;
                    targets.push_back(probability_matrix);
                }
            } else if (winner == BoardSquare::O) {
                for (const auto& b : player_o_game_history) inputs.push_back(b);
                for (int move : player_o_moves) {
                    Eigen::MatrixXf probability_matrix = Eigen::MatrixXf::Zero(9, 1);
                    probability_matrix(move) = 1.0f;
                    targets.push_back(probability_matrix);
                }
            } else {
                // if its a draw then train both players moves as good ones
                num_draws++;
                for (const auto& b : player_x_game_history) inputs.push_back(b);
                for (int move : player_x_moves) {
                    Eigen::MatrixXf probability_matrix = Eigen::MatrixXf::Zero(9, 1);
                    probability_matrix(move) = 1.0f;
                    targets.push_back(probability_matrix);
                }
                for (const auto& b : player_o_game_history) inputs.push_back(b);
                for (int move : player_o_moves) {
                    Eigen::MatrixXf probability_matrix = Eigen::MatrixXf::Zero(9, 1);
                    probability_matrix(move) = 1.0f;
                    targets.push_back(probability_matrix);
                }
            }
        }

        std::cout << "epoch " << epoch << " draw ratio: " << ((float)num_draws) / num_games << " epsilon: " << epsilon << std::endl;
        
        // train on the data
        Eigen::MatrixXf minibatch_inputs = Eigen::MatrixXf::Zero(9, 1000);
        Eigen::MatrixXf minibatch_targets = Eigen::MatrixXf::Zero(9, 1000);
        for (int i = 0; i < num_batch_training; i++) {
            nn_utils::get_batch(inputs, targets, minibatch_inputs, minibatch_targets, minibatch_size);
            optimiser.step(minibatch_inputs, minibatch_targets);
        }

        epsilon *= epsilon_decay;
    }

    ffnn.write_to_file("tictactoe/network.dat");
}

int main() {
    naive();
    return 0;
}