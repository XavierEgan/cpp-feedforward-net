#pragma once
#include "FFNN.hpp"
#include "Workspace.hpp"
#include "NN_Utils.hpp"

#include <cmath>

struct AdamOptimiser {
    FFNN& ffnn;
    CostType cost_type;
    RegularizationType reg_type;
    float reg_lambda;
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    int t = 1;
    BackpropWorkspace ws;
    std::vector<Eigen::MatrixXf> weight_m, weight_v, bias_m, bias_v;

    static AdamOptimiser from_ffnn(FFNN& ffnn, CostType cost_type, float lr = 0.001f, RegularizationType reg_type = RegularizationType::none, float reg_lambda = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f) {
        AdamOptimiser opt(ffnn, cost_type, reg_type, reg_lambda, lr, beta1, beta2, epsilon);

        const size_t depth = ffnn.depth();
        opt.weight_m.resize(depth - 1);
        opt.weight_v.resize(depth - 1);
        opt.bias_m.resize(depth - 1);
        opt.bias_v.resize(depth - 1);

        for (size_t l = 1; l < depth; l++) {
            opt.weight_m.at(l - 1) = Eigen::MatrixXf::Zero(ffnn.get_weight(l).rows(), ffnn.get_weight(l).cols());
            opt.weight_v.at(l - 1) = Eigen::MatrixXf::Zero(ffnn.get_weight(l).rows(), ffnn.get_weight(l).cols());
            opt.bias_m.at(l - 1) = Eigen::MatrixXf::Zero(ffnn.get_bias(l).rows(), 1);
            opt.bias_v.at(l - 1) = Eigen::MatrixXf::Zero(ffnn.get_bias(l).rows(), 1);
        }

        return opt;
    }

    /*
    do a learning step and return the average cost over the minibatch
    */
    float step(const Eigen::MatrixXf& minibatch, const Eigen::MatrixXf& minibatch_targets) {
        if (minibatch.cols() <= 0) {
            throw std::invalid_argument("step: minibatch size must be non-zero");
        }

        const float cost = ffnn.backward_and_loss(minibatch, minibatch_targets, cost_type, ws);
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

        ws.delta.at(ffnn.depth() - 2) = output_delta;
        ffnn.backward(minibatch, ws);
        apply_gradients();
    }

    AdamOptimiser() = delete;
private:
    AdamOptimiser(FFNN& ffnn, CostType cost_type, RegularizationType reg_type, float reg_lambda, float lr, float beta1, float beta2, float epsilon)
        : ffnn(ffnn), cost_type(cost_type), reg_type(reg_type), reg_lambda(reg_lambda), lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), ws(BackpropWorkspace::from_shape(ffnn.network_shape)) {}

    void adam_update(Eigen::MatrixXf& param, const Eigen::MatrixXf& grad, Eigen::MatrixXf& m, Eigen::MatrixXf& v, float alpha_t, float eps_hat) {
        m = beta1 * m + (1.0f - beta1) * grad;
        v = beta2 * v + (1.0f - beta2) * grad.cwiseAbs2();
        param.array() -= alpha_t * m.array() / (v.array().sqrt() + eps_hat);
    }

    // consumes ws.weight_grad/bias_grad (already filled by whichever backward variant was
    // called) and applies the Adam update to every layer
    void apply_gradients() {
        // bias-correction folded into per-step scalars so no m_hat/v_hat temporaries are needed
        const float alpha_t = lr * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));
        const float eps_hat = epsilon * std::sqrt(1.0f - std::pow(beta2, t));

        for (size_t l = 1; l < ffnn.depth(); l++) {
            nn_utils::add_regularization(ws.weight_grad.at(l - 1), ffnn.get_weight(l), reg_type, reg_lambda);

            adam_update(ffnn.get_weight(l), ws.weight_grad.at(l - 1), weight_m.at(l - 1), weight_v.at(l - 1), alpha_t, eps_hat);
            adam_update(ffnn.get_bias(l), ws.bias_grad.at(l - 1), bias_m.at(l - 1), bias_v.at(l - 1), alpha_t, eps_hat);
        }

        t++;
    }
};
