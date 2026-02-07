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
#include <fstream>

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

        ForwardResult forward_result = ForwardResult(network_shape);
        BackpropResult backprop_result = BackpropResult(network_shape);

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

    static FFNN from_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("from_file: could not open file");
        }

        // read len of network_shape
        size_t network_shape_len;
        file.read(reinterpret_cast<char*>(&network_shape_len), sizeof(size_t));

        // read network_shape
        std::vector<size_t> network_shape(network_shape_len);
        file.read(reinterpret_cast<char*>(network_shape.data()), network_shape_len * sizeof(size_t));

        // read activation functions
        std::vector<ActivationFunc> activation_funcs(network_shape_len - 1);
        file.read(reinterpret_cast<char*>(activation_funcs.data()), activation_funcs.size() * sizeof(ActivationFunc));

        ForwardResult forward_result = ForwardResult(network_shape);
        BackpropResult backprop_result = BackpropResult(network_shape);
        FFNN ffnn(forward_result, backprop_result);
        ffnn.depth = network_shape_len;
        ffnn.network_shape = network_shape;
        ffnn.activation_functions = activation_funcs;

        // read weights and biases
        for (size_t l = 1; l < ffnn.depth; l++) {
            Eigen::MatrixXf weight(network_shape.at(l), network_shape.at(l - 1));
            Eigen::MatrixXf bias(network_shape.at(l), 1);

            file.read(reinterpret_cast<char*>(weight.data()), weight.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float));

            ffnn.weights.push_back(weight);
            ffnn.biases.push_back(bias);
        }
        
        return ffnn;
    }

    const Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) {
        forward_result.set_a(0, input);

        for (size_t l = 1; l < depth; l++) {
            Eigen::MatrixXf z = (get_weight(l) * forward_result.get_a(l - 1)).colwise() + get_bias(l).col(0);
            forward_result.set_z(l, z);
            forward_result.set_a(l, nn_utils::activate(z, get_activation_func(l)));
        }

        return forward_result.get_a(depth - 1);
    }

    void backwards(const Eigen::MatrixXf& input, const Eigen::MatrixXf& target, CostType cost_type) {
        if (cost_type == CostType::binary_cross_entropy && activation_functions.back() != ActivationFunc::sigmoid) {
            throw std::invalid_argument("from_random: binary_cross_entropy cost requires sigmoid activation in output layer");
        }

        if (cost_type == CostType::categorical_cross_entropy && activation_functions.back() != ActivationFunc::softmax) {
            throw std::invalid_argument("from_random: categorical_cross_entropy cost requires softmax activation in output layer");
        }

        forward(input);

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

    void write_to_file(const std::string& filename) const {
        // file format (in binary):
        // length of network_shape
        // input_size layer1_size layer2_size ... layerN_size
        // activation_func1 activation_func2 ... activation_funcN-1
        // weight matrix 1 (row major, values separated by spaces)
        // bias vector 1 (values separated by spaces)
        // weight matrix 2
        // bias vector 2
        // ...
        // weight matrix N-1
        // bias vector N-1

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("write_to_file: could not open file");
        }

        // write text length of network_shape
        size_t network_shape_len = network_shape.size();
        file.write(reinterpret_cast<const char*>(&network_shape_len), sizeof(size_t));

        // write layer sizes
        file.write(reinterpret_cast<const char*>(network_shape.data()), network_shape.size() * sizeof(size_t));

        // write activation functions
        file.write(reinterpret_cast<const char*>(activation_functions.data()), activation_functions.size() * sizeof(ActivationFunc));

        // write weights and biases
        for (size_t l = 1; l < depth; l++) {
            const Eigen::MatrixXf& weight = get_weight(l);
            const Eigen::MatrixXf& bias = get_bias(l);

            file.write(reinterpret_cast<const char*>(weight.data()), weight.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(float));
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