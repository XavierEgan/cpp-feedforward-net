#pragma once
#include "FFNN.hpp"

struct GradientDescentOptimiser {
    FFNN& ffnn;
    CostType cost_type;
    double lr;
    
    GradientDescentOptimiser(FFNN& ffnn, CostType cost_type, double learning_rate = 0.005) : ffnn(ffnn), cost_type(cost_type), lr(learning_rate) {}

    /*
    Do a learning step and return the average cost over the minibatch
    */
    float step(const Eigen::MatrixXf& minibatch, const Eigen::MatrixXf& minibatch_targets) {
        if (minibatch.cols() <= 0) {
            throw std::invalid_argument("gradient_descent: minibatch size must be non-zero");
        }

        // backprop to get gradients
        ffnn.backwards(minibatch, minibatch_targets, cost_type);

        // update weights and biases
        for (size_t l = 1; l < ffnn.depth; l++) {
            ffnn.get_weight(l) -= lr * ffnn.backprop_result.get_weight_gradient(l);
            ffnn.get_bias(l) -= lr * ffnn.backprop_result.get_bias_gradient(l);
        }

        return ffnn.backprop_result.get_cost();
    }
};