#pragma once
#include "Eigen/Dense"
#include "NN_Utils.hpp"

#include <vector>


struct BackpropResult {
    BackpropResult() = delete;
    BackpropResult(int depth, std::vector<size_t> network_shape) : cost(0.0f), depth(depth) {
        weight_gradients.resize(depth - 1);
        bias_gradients.resize(depth - 1);

        for (int l = 1; l < depth; l++) {
            set_weight_gradient(l , Eigen::MatrixXf::Zero(network_shape.at(l), network_shape.at(l - 1)));
            set_bias_gradient(l , Eigen::MatrixXf::Zero(network_shape.at(l), 1));
        }
    }

    void set_weight_gradient(int l, const Eigen::MatrixXf& grad) {
        nn_utils::check_layer_in_range(l, depth);
        weight_gradients.at(l - 1) = grad;
    }
    void set_bias_gradient(int l, const Eigen::MatrixXf& grad) {
        nn_utils::check_layer_in_range(l, depth);
        bias_gradients.at(l - 1) = grad;
    }

    const Eigen::MatrixXf& get_weight_gradient(int l) const {
        nn_utils::check_layer_in_range(l, depth);
        return weight_gradients.at(l - 1);
    }
    const Eigen::MatrixXf& get_bias_gradient(int l) const {
        nn_utils::check_layer_in_range(l, depth);
        return bias_gradients.at(l - 1);
    }

    float& get_cost() {
        return cost;
    }

private:
    float cost;
    int depth;
    std::vector<Eigen::MatrixXf> weight_gradients;
    std::vector<Eigen::MatrixXf> bias_gradients;
};
