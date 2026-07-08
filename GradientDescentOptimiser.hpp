#pragma once
#include "FFNN.hpp"
#include "Workspace.hpp"
#include "NN_Utils.hpp"

struct GradientDescentOptimiser {
    FFNN& ffnn;
    CostType cost_type;
    RegularizationType reg_type;
    float reg_lambda;
    float lr;
    BackpropWorkspace ws;

    static GradientDescentOptimiser from_ffnn(FFNN& ffnn, CostType cost_type, float lr = 0.005f, RegularizationType reg_type = RegularizationType::none, float reg_lambda = 1e-3f) {
        return GradientDescentOptimiser(ffnn, cost_type, reg_type, reg_lambda, lr);
    }

    /*
    do a learning step and return the average cost over the minibatch
    */
    float step(const Eigen::MatrixXf& minibatch, const Eigen::MatrixXf& minibatch_targets) {
        if (minibatch.cols() <= 0) {
            throw std::invalid_argument("step: minibatch size must be non-zero");
        }

        const float cost = ffnn.backward(minibatch, minibatch_targets, cost_type, ws);

        for (size_t l = 1; l < ffnn.depth(); l++) {
            nn_utils::add_regularization(ws.weight_grad.at(l - 1), ffnn.get_weight(l), reg_type, reg_lambda);

            ffnn.get_weight(l) -= lr * ws.weight_grad.at(l - 1);
            ffnn.get_bias(l) -= lr * ws.bias_grad.at(l - 1);
        }

        return cost;
    }

    GradientDescentOptimiser() = delete;
private:
    GradientDescentOptimiser(FFNN& ffnn, CostType cost_type, RegularizationType reg_type, float reg_lambda, float lr)
        : ffnn(ffnn), cost_type(cost_type), reg_type(reg_type), reg_lambda(reg_lambda), lr(lr), ws(BackpropWorkspace::from_shape(ffnn.network_shape)) {}
};
