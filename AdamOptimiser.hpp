#pragma once
#include "FFNN.hpp"
#include "AdamBuffer.hpp"

struct AdamOptimiser {
    FFNN& ffnn;
    AdamBuffer adam_buffer;
    CostType cost_type;
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t = 1;

    AdamOptimiser(FFNN& ffnn, CostType cost_type, double learning_rate = 0.01, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) : ffnn(ffnn), adam_buffer(AdamBuffer(ffnn.depth, ffnn.network_shape)), cost_type(cost_type), lr(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    /*
    Do a learning step and return the average cost over the minibatch
    */
    float step(const Eigen::MatrixXf& minibatch, const Eigen::MatrixXf& minibatch_targets) {
        const int minibatch_size = minibatch.cols();

        if (minibatch_size <= 0) {
            throw std::invalid_argument("gradient_descent: minibatch size must be non-zero");
        }

        // backprop to get gradients
        ffnn.backwards(minibatch, minibatch_targets, cost_type);
        
        // update weights and biases
        for (size_t l = 1; l < ffnn.depth; l++) {
            adam_buffer.set_weight_m(l, beta1 * adam_buffer.get_weight_m(l) + (1.0 - beta1) * ffnn.backprop_result.get_weight_gradient(l));
            adam_buffer.set_weight_v(l, beta2 * adam_buffer.get_weight_v(l) + (1.0 - beta2) * ffnn.backprop_result.get_weight_gradient(l).cwiseSquare());

            Eigen::MatrixXf m_hat = adam_buffer.get_weight_m(l) / (1.0 - std::pow(beta1, t));
            Eigen::MatrixXf v_hat = adam_buffer.get_weight_v(l) / (1.0 - std::pow(beta2, t));

            ffnn.get_weight(l) -= lr * m_hat.cwiseQuotient((v_hat.array().sqrt() + epsilon).matrix());

            adam_buffer.set_bias_m(l, beta1 * adam_buffer.get_bias_m(l) + (1.0 - beta1) * ffnn.backprop_result.get_bias_gradient(l));
            adam_buffer.set_bias_v(l, beta2 * adam_buffer.get_bias_v(l) + (1.0 - beta2) * ffnn.backprop_result.get_bias_gradient(l).cwiseSquare());

            Eigen::MatrixXf m_hat_b = adam_buffer.get_bias_m(l) / (1.0 - std::pow(beta1, t));
            Eigen::MatrixXf v_hat_b = adam_buffer.get_bias_v(l) / (1.0 - std::pow(beta2, t));

            ffnn.get_bias(l) -= lr * m_hat_b.cwiseQuotient((v_hat_b.array().sqrt() + epsilon).matrix());
        }

        t++;

        return ffnn.backprop_result.get_cost();
    }
};