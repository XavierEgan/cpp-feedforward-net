#pragma once
#include "Eigen/Dense"
#include "NN_Utils.hpp"
#include <stdexcept>

enum CostType {
    quadratic,
    binary_cross_entropy,
    categorical_cross_entropy
};

namespace nn_utils {
float cost(const Eigen::MatrixXf& a, const Eigen::MatrixXf& y, CostType cost_type) {
    //std::cout << a << std::endl << std::endl;
    switch (cost_type) {
        case quadratic: {
            return ((a - y).cwiseSquare() * 0.5).sum() / (a.rows() * a.cols());
        }
        case binary_cross_entropy: {
            Eigen::ArrayXXf ac = a.array().min(1.0f - k_prob_eps).max(k_prob_eps);
            return (-(y.array() * ac.log()) - (1.0f - y.array()) * (1.0f - ac).log()).sum() / (a.rows() * a.cols());
        }
        case categorical_cross_entropy: {
            Eigen::ArrayXXf ac = a.array().min(1.0f - k_prob_eps).max(k_prob_eps);
            return (-(y.array() * ac.log())).sum() / (a.rows() * a.cols());
        }
        default:
            throw std::invalid_argument("cost: unknown cost type");
    }
}

Eigen::MatrixXf cost_derivative(const Eigen::MatrixXf& a, const Eigen::MatrixXf& y, CostType cost_type) {
    switch (cost_type) {
        case quadratic:
            return a - y;
        case binary_cross_entropy: {
            Eigen::ArrayXXf ac = a.array().min(1.0f - k_prob_eps).max(k_prob_eps);
            return (-(y.array() / ac) + (1.0f - y.array()) / (1.0f - ac)).matrix();
        }
        // categorical cross-entropy and binary cross-entropy are guarenteed to be simplified because they are always paired with softmax and sigmoid respectively
        default:
            throw std::invalid_argument("cost_derivative: unknown cost type");
    }
}
}