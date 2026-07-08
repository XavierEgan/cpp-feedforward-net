#pragma once
#include "Eigen/Dense"
#include "RegularizationType.hpp"

#include <string>

namespace nn_utils {

constexpr float k_prob_eps = 1e-7f;

template<typename T>
std::string get_matrix_shape_str(const T& m) {
    return "(" + std::to_string(m.rows()) + ", " + std::to_string(m.cols()) + ")";
}

// adds the regularization penalty to an already-averaged weight gradient (biases are never
// regularized); not divided by batch size, matching the recorded hyperparameter history
inline void add_regularization(Eigen::MatrixXf& grad, const Eigen::MatrixXf& weight, RegularizationType reg_type, float reg_lambda) {
    if (reg_type == RegularizationType::l2) {
        grad += reg_lambda * weight;
    } else if (reg_type == RegularizationType::l1) {
        grad += reg_lambda * weight.cwiseSign();
    }
}

}