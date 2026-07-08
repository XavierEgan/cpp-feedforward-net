#pragma once
#include "Eigen/Dense"
#include "RegularizationType.hpp"

#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>

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

inline void get_random_batch(const std::vector<Eigen::MatrixXf>& inputs, const std::vector<Eigen::MatrixXf>& targets, Eigen::MatrixXf& minibatch, Eigen::MatrixXf& minibatch_targets, int batch_size = -1, unsigned int seed = std::random_device{}()) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("get_random_batch: inputs and targets must be the same size");
    }
    if (inputs.size() == 0) {
        throw std::invalid_argument("get_random_batch: inputs and targets cannot be empty");
    }

    const int n =  inputs.size();

    if (batch_size == -1) {
        minibatch.resize(inputs.at(0).rows(), n);
        minibatch_targets.resize(targets.at(0).rows(), n);
    }

    if (batch_size <= 0) batch_size = n;
    if (batch_size > n) batch_size = n;

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    static thread_local std::mt19937 rng(seed);

    if (batch_size < n) {
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(batch_size);
    }

    minibatch.resize(inputs.at(0).rows(), batch_size);
    minibatch_targets.resize(targets.at(0).rows(), batch_size);

    for (int b = 0; b < batch_size; b++) {
        minibatch.col(b) = inputs.at(indices.at(b));
        minibatch_targets.col(b) = targets.at(indices.at(b));
    }
}

}