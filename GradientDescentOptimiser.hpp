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
        apply_gradients();

        return cost;
    }

    /*
    like step(), but takes the output layer's error directly instead of deriving it from a
    target/cost_type - e.g. a gradient backpropagated in from a downstream network (a frozen
    critic/discriminator). requires ws.fwd to already hold minibatch's forward pass (call
    ffnn.forward(minibatch, ws.fwd) first, e.g. as part of computing output_delta upstream).
    does not return a cost since none is computed here.
    */
    void step_from_output_delta(const Eigen::MatrixXf& minibatch, const Eigen::MatrixXf& output_delta) {
        if (minibatch.cols() <= 0) {
            throw std::invalid_argument("step_from_output_delta: minibatch size must be non-zero");
        }

        ffnn.backward_from_output_delta(minibatch, output_delta, ws);
        apply_gradients();
    }

    GradientDescentOptimiser() = delete;
private:
    GradientDescentOptimiser(FFNN& ffnn, CostType cost_type, RegularizationType reg_type, float reg_lambda, float lr)
        : ffnn(ffnn), cost_type(cost_type), reg_type(reg_type), reg_lambda(reg_lambda), lr(lr), ws(BackpropWorkspace::from_shape(ffnn.network_shape)) {}

    // consumes ws.weight_grad/bias_grad (already filled by whichever backward variant was
    // called) and applies the gradient step to every layer
    void apply_gradients() {
        for (size_t l = 1; l < ffnn.depth(); l++) {
            nn_utils::add_regularization(ws.weight_grad.at(l - 1), ffnn.get_weight(l), reg_type, reg_lambda);

            ffnn.get_weight(l) -= lr * ws.weight_grad.at(l - 1);
            ffnn.get_bias(l) -= lr * ws.bias_grad.at(l - 1);
        }
    }
};
