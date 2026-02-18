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
#include <immintrin.h>
void naive() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    const int board_size = 4;

    std::vector<size_t> network_shape {board_size * board_size, 32, 32, board_size * board_size};
    std::vector<ActivationFunc> activation_funcs {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax};

    FFNN ffnn = FFNN::from_random_he_scaling(network_shape, activation_funcs);
    //FFNN ffnn = FFNN::from_file("tictactoe/network_6000.dat");

    int num_epochs = 10000;

    int num_self_games = 1000;
    int num_random_games = 500;
    double epsilon = 0.2;
    double epsilon_decay = 0.9998;

    int minibatch_size = -1;
    int num_batch_training = 25;

    TicTacToe<board_size> tic_tac_toe;
    AdamOptimiser optimiser(ffnn, CostType::categorical_cross_entropy);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        if (epoch % 1000 == 0 && epoch != 0) {
            std::cout << "Saving network at epoch " << epoch << std::endl;
            ffnn.write_to_file("tictactoe/network_" + std::to_string(epoch) + ".dat");
        }

        std::vector<Eigen::MatrixXf> inputs;
        std::vector<Eigen::MatrixXf> targets;

        // self play to generate training data
        for (int game = 0; game < num_self_games; game++) {
            tic_tac_toe.restart();

            BoardSquare winner = BoardSquare::EMPTY;

            // if we do board_size * board_size iterations and nobody one then the game is a draw
            for (int i = 0; i < board_size * board_size; i++) {
                // get board state
                Eigen::MatrixXf board_state = tic_tac_toe.get_board_state(tic_tac_toe.next_player);

                // get network prediction
                Eigen::MatrixXf move_probabilities = ffnn.forward(board_state);
                
                // play the move
                tic_tac_toe.play_move(move_probabilities, epsilon);
                
                winner = tic_tac_toe.check_winner();
                if (winner != BoardSquare::EMPTY) {
                    break;
                };
            }
            
            if (winner == BoardSquare::X) {
                tic_tac_toe.append_player_x_positive_example(inputs, targets);
            } else if (winner == BoardSquare::O) {
                tic_tac_toe.append_player_o_positive_example(inputs, targets);
            }
        }

        // play against an agent that plays randomly
        int num_random_agent_wins = 0;
        int num_our_agent_wins = 0;
        int num_draws = 0;
        for (int game = 0; game < num_random_games; game++) {
            tic_tac_toe.restart();

            BoardSquare winner = BoardSquare::EMPTY;

            BoardSquare our_agent = game % 2 == 0 ? BoardSquare::X : BoardSquare::O;
            BoardSquare random_agent = our_agent == BoardSquare::X ? BoardSquare::O : BoardSquare::X;

            for (int i = 0; i < board_size * board_size; i++) {
                if (tic_tac_toe.next_player == our_agent) {
                    Eigen::MatrixXf board_state = tic_tac_toe.get_board_state(our_agent);
                    Eigen::MatrixXf move_probabilities = ffnn.forward(board_state);
                    tic_tac_toe.play_move(move_probabilities, epsilon);
                } else {
                    tic_tac_toe.play_move(Eigen::MatrixXf(), 1.0f); // play random move
                }
                
                winner = tic_tac_toe.check_winner();
                if (winner != BoardSquare::EMPTY) {
                    break;
                };
            }

            if (winner == our_agent) {
                tic_tac_toe.append_player_positive_example(inputs, targets, our_agent);
                num_our_agent_wins++;
            } else if (winner == random_agent) {
                num_random_agent_wins++;
            } else {
                num_draws++;
            }
        }

        std::cout << "Epoch " << epoch 
            << " Win Ratio: " << num_our_agent_wins / (float)num_random_games
            << " Draws: " << num_draws
            << " Loses: " << num_random_agent_wins
            << " epsilon: " << epsilon << std::endl;


        // train on the data
        Eigen::MatrixXf minibatch_inputs;
        Eigen::MatrixXf minibatch_targets;
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