#pragma once

#include "Eigen/Dense"
#include "ForwardResult.hpp"
#include "BackpropResult.hpp"
#include "CostType.hpp"
#include "ActivationFunction.hpp"

#include <vector>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>

struct FFNN {
    size_t depth;
    std::vector<size_t> network_shape;
    std::vector<ActivationFunc> activation_functions;
    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::MatrixXf> biases;

    ForwardResult forward_result;
    BackpropResult backprop_result;

    static FFNN from_random_he_scaling(const std::vector<size_t>& network_shape, const std::vector<ActivationFunc>& activation_funcs) {
        size_t depth = network_shape.size();

        if (network_shape.size() != activation_funcs.size() + 1) {
            throw std::invalid_argument("from_random: network_shape.size() must equal activation_funcs.size() + 1");
        }

        if (network_shape.size() < 2) {
            throw std::invalid_argument("from_random: network_shape must have at least 2 layers");
        }

        for (size_t i = 0; i < network_shape.size(); i++) {
            if (network_shape[i] == 0) {
                throw std::invalid_argument("from_random: network_shape entries must be > 0");
            }
        }

        ForwardResult forward_result = ForwardResult(depth, network_shape);
        BackpropResult backprop_result = BackpropResult(depth, network_shape);

        FFNN ffnn(forward_result, backprop_result);

        ffnn.depth = depth;
        ffnn.activation_functions = activation_funcs;
        ffnn.network_shape = network_shape;

        // weights = output x input
        // He-style scaling
        for (size_t i = 1; i < ffnn.depth; i++) {
            const float fan_in = static_cast<float>(network_shape.at(i - 1));
            const float scale = std::sqrt(2.0f / fan_in);
            ffnn.weights.push_back(Eigen::MatrixXf::Random(network_shape.at(i), network_shape.at(i - 1)) * scale);
            ffnn.biases.push_back(Eigen::MatrixXf::Zero(network_shape.at(i), 1));
        }

        return ffnn;
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        Eigen::MatrixXf prev_a = input;
        for (size_t l = 1; l < depth; l++) {
            
            Eigen::MatrixXf z = (get_weight(l) * prev_a).colwise() + get_bias(l).col(0);
            prev_a = nn_utils::activate(z, get_activation_func(l));
        }
        return prev_a;
    }

    void forward(ForwardResult& forward_result) {
        for (size_t l = 1; l < depth; l++) {
            Eigen::MatrixXf z = (get_weight(l) * forward_result.get_a(l - 1)).colwise() + get_bias(l).col(0);
            forward_result.set_z(l, z);
            forward_result.set_a(l, nn_utils::activate(z, get_activation_func(l)));
        }
    }

    void backwards(const Eigen::MatrixXf& input, const Eigen::MatrixXf& target, CostType cost_type) {
        if (cost_type == CostType::binary_cross_entropy && activation_functions.back() != ActivationFunc::sigmoid) {
            throw std::invalid_argument("from_random: binary_cross_entropy cost requires sigmoid activation in output layer");
        }

        if (cost_type == CostType::categorical_cross_entropy && activation_functions.back() != ActivationFunc::softmax) {
            throw std::invalid_argument("from_random: categorical_cross_entropy cost requires softmax activation in output layer");
        }

        forward_result.set_a(0, input);
        forward(forward_result);

        backprop_result.get_cost() = nn_utils::cost(forward_result.get_a(depth - 1), target, cost_type);

        // get errors in the output layer
        Eigen::MatrixXf output_error;
        if (cost_type == CostType::binary_cross_entropy || cost_type == CostType::categorical_cross_entropy) {
            // special case where the derivative simplifies
            output_error = forward_result.get_a(depth - 1) - target;
        } else {
            output_error = nn_utils::cost_derivative(forward_result.get_a(depth - 1), target, cost_type).cwiseProduct(nn_utils::activate_der(forward_result.get_z(depth - 1), get_activation_func(depth - 1)));
        }
        Eigen::MatrixXf prev_error = output_error;

        backprop_result.set_weight_gradient(depth - 1, output_error * forward_result.get_a(depth - 2).transpose() / input.cols());
        backprop_result.set_bias_gradient(depth - 1, output_error.rowwise().mean());
        
        // get error in all future layers
        for (size_t l = depth - 2; l > 0; l--) {
            // weight connecting this layer to the next
            // the error in the next layer
            // and the derivative of the activation function of this layer
            Eigen::MatrixXf this_error = (get_weight(l + 1).transpose() * prev_error).cwiseProduct(nn_utils::activate_der(forward_result.get_z(l), get_activation_func(l)));
            
            backprop_result.set_weight_gradient(l, this_error * forward_result.get_a(l - 1).transpose() / input.cols());
            backprop_result.set_bias_gradient(l, this_error.rowwise().mean());
            
            prev_error = this_error;
        }
    }

    // gets the weight matrix connecting l-1 to l
    Eigen::MatrixXf& get_weight(size_t l) {
        if (l < 1 || l >= depth) {
            throw std::out_of_range("get_weight: l is out of range" + std::to_string(l));
        }
        return weights.at(l - 1);
    }
    const Eigen::MatrixXf& get_weight(size_t l) const {
        if (l < 1 || l >= depth) {
            throw std::out_of_range("get_weight (const): l is out of range");
        }
        return weights.at(l - 1);
    }

    // gets the biases of layer l
    Eigen::MatrixXf& get_bias(size_t l) {
        if (l < 1 || l >= depth) {
            throw std::out_of_range("get_bias: l is out of range");
        }
        return biases.at(l - 1);
    }
    const Eigen::MatrixXf& get_bias(size_t l) const {
        if (l < 1 || l >= depth) {
            throw std::out_of_range("get_bias (const): l is out of range");
        }
        return biases.at(l - 1);
    }

    // gets the activation function of layer l
    ActivationFunc get_activation_func(size_t l) const {
        if (l < 1 || l >= depth) {
            throw std::out_of_range("get_activation_func: l is out of range" + std::to_string(l));
        }
        return activation_functions.at(l - 1);
    }

    FFNN() = delete;
private: 
    FFNN(ForwardResult forward_result, BackpropResult backprop_result) : forward_result(forward_result), backprop_result(backprop_result) {}
};