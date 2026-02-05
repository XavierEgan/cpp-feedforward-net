#pragma once
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>

namespace nn_utils {

constexpr float kProbEps = 1e-7f;

template<typename T>
std::string get_matrix_shape_str(const T& m) {
    return "(" + std::to_string(m.rows()) + ", " + std::to_string(m.cols()) + ")";
}

void check_layer_in_range(int l, int depth) {
    if (l < 1 || l >= depth) {
        throw std::out_of_range("get_z: l is out of range: " + std::to_string(l));
    }
}

void check_matrix_shape(const Eigen::MatrixXf& m, int rows, int cols, const std::string& name) {
    if (m.rows() != rows || m.cols() != cols) {
        throw std::invalid_argument(name + " has incorrect shape. Expected (" + std::to_string(rows) + ", " + std::to_string(cols) + "), got " + get_matrix_shape_str(m));
    }
}

void get_batch(const std::vector<Eigen::MatrixXf>& inputs, const std::vector<Eigen::MatrixXf>& targets, Eigen::MatrixXf& minibatch, Eigen::MatrixXf& minibatch_targets, int batch_size = -1) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("get_batch: inputs and targets must be the same size");
    }
    if (inputs.size() == 0) {
        throw std::invalid_argument("get_batch: inputs and targets cannot be empty");
    }

    const int n =  inputs.size();

    if (batch_size <= 0) batch_size = n;
    if (batch_size > n) batch_size = n;

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    static thread_local std::mt19937 rng(std::random_device{}());

    if (batch_size < n) {
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(batch_size);
    }

    minibatch.resize(inputs.at(0).rows(), batch_size);
    minibatch_targets.resize(targets.at(0).rows(), batch_size);

    for (int b = 0; b < batch_size; b++) {
        // std::cout << "here" << std::endl;

        // std::cout << "Minibatch size: " << get_matrix_shape_str(minibatch) << std::endl;

        // std::cout << "LHS Shape: " << get_matrix_shape_str(minibatch.col(b)) << ", RHS Shape: " << get_matrix_shape_str(inputs.at(indices.at(b)).col(0)) << std::endl;
        
        minibatch.col(b) = inputs.at(indices.at(b));
        minibatch_targets.col(b) = targets.at(indices.at(b));
    }
}

}