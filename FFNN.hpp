#pragma once

#include "Eigen/Dense"
#include "Workspace.hpp"
#include "CostType.hpp"
#include "ActivationFunction.hpp"

#include <vector>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>

// holds only the trained parameters of a feedforward network
// forward/backward are const and write into a caller-owned workspace, so a single FFNN can be
// shared across threads or evaluated repeatedly without reallocating
struct FFNN {
    std::vector<size_t> network_shape;
    std::vector<ActivationFunc> activation_functions;
    std::vector<Eigen::MatrixXf> weights;   // weights.at(l - 1): shape[l] x shape[l - 1]
    std::vector<Eigen::MatrixXf> biases;    // biases.at(l - 1): shape[l] x 1

    size_t depth() const {
        return network_shape.size();
    }

    static FFNN from_random_he_scaling(const std::vector<size_t>& network_shape, const std::vector<ActivationFunc>& activation_funcs) {
        if (network_shape.size() != activation_funcs.size() + 1) {
            throw std::invalid_argument("from_random_he_scaling: network_shape.size() must equal activation_funcs.size() + 1");
        }

        if (network_shape.size() < 2) {
            throw std::invalid_argument("from_random_he_scaling: network_shape must have at least 2 layers");
        }

        for (size_t i = 0; i < network_shape.size(); i++) {
            if (network_shape.at(i) == 0) {
                throw std::invalid_argument("from_random_he_scaling: network_shape entries must be > 0");
            }
        }

        FFNN ffnn(network_shape, activation_funcs);

        // weights = output x input
        // He-style scaling
        for (size_t i = 1; i < ffnn.depth(); i++) {
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
            throw std::runtime_error("from_file: could not open file " + filename);
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

        FFNN ffnn(network_shape, activation_funcs);

        // read weights and biases
        for (size_t l = 1; l < ffnn.depth(); l++) {
            Eigen::MatrixXf weight(network_shape.at(l), network_shape.at(l - 1));
            Eigen::MatrixXf bias(network_shape.at(l), 1);

            file.read(reinterpret_cast<char*>(weight.data()), weight.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float));

            ffnn.weights.push_back(weight);
            ffnn.biases.push_back(bias);
        }

        return ffnn;
    }

    // allocation-free after the first call with a given batch size, returns a reference into ws
    const Eigen::MatrixXf& forward(const Eigen::MatrixXf& input, ForwardWorkspace& ws) const {
        const Eigen::MatrixXf* prev_a = &input;

        for (size_t l = 1; l < depth(); l++) {
            Eigen::MatrixXf& z = ws.z.at(l - 1);
            z.noalias() = get_weight(l) * (*prev_a);
            z.colwise() += get_bias(l).col(0);

            nn_utils::activate(z, get_activation_func(l), ws.a.at(l - 1));
            prev_a = &ws.a.at(l - 1);
        }

        return ws.a.at(depth() - 2);
    }

    // convenience one-off inference: allocates its own workspace, returns a copy of the result
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) const {
        ForwardWorkspace ws = ForwardWorkspace::from_shape(network_shape);
        return forward(input, ws);
    }

    // runs forward, fills ws.weight_grad / ws.bias_grad, and returns the average cost over the batch
    float backward(const Eigen::MatrixXf& input, const Eigen::MatrixXf& target, CostType cost_type, BackpropWorkspace& ws) const {
        if (cost_type == CostType::binary_cross_entropy && activation_functions.back() != ActivationFunc::sigmoid) {
            throw std::invalid_argument("backward: binary_cross_entropy cost requires sigmoid activation in output layer");
        }

        if (cost_type == CostType::categorical_cross_entropy && activation_functions.back() != ActivationFunc::softmax) {
            throw std::invalid_argument("backward: categorical_cross_entropy cost requires softmax activation in output layer");
        }

        const size_t d = depth();
        const Eigen::MatrixXf& output = forward(input, ws.fwd);
        const float total_cost = nn_utils::cost(output, target, cost_type);

        // activation of the layer feeding into layer l (the input itself for layer 1)
        auto prev_activation = [&](size_t l) -> const Eigen::MatrixXf& {
            return l == 1 ? input : ws.fwd.a.at(l - 2);
        };

        // error in the output layer
        if (cost_type == CostType::binary_cross_entropy || cost_type == CostType::categorical_cross_entropy) {
            // special case where the derivative simplifies
            ws.delta.at(d - 2) = output - target;
        } else {
            nn_utils::activate_der(ws.fwd.z.at(d - 2), get_activation_func(d - 1), ws.activation_der.at(d - 2));
            ws.delta.at(d - 2) = nn_utils::cost_derivative(output, target, cost_type).cwiseProduct(ws.activation_der.at(d - 2));
        }

        ws.weight_grad.at(d - 2).noalias() = ws.delta.at(d - 2) * prev_activation(d - 1).transpose() / input.cols();
        ws.bias_grad.at(d - 2) = ws.delta.at(d - 2).rowwise().mean();

        // error in all earlier layers
        for (size_t l = d - 2; l > 0; l--) {
            // weight connecting this layer to the next, the error in the next layer,
            // and the derivative of the activation function of this layer
            nn_utils::activate_der(ws.fwd.z.at(l - 1), get_activation_func(l), ws.activation_der.at(l - 1));
            ws.delta.at(l - 1).noalias() = get_weight(l + 1).transpose() * ws.delta.at(l);
            ws.delta.at(l - 1).array() *= ws.activation_der.at(l - 1).array();

            ws.weight_grad.at(l - 1).noalias() = ws.delta.at(l - 1) * prev_activation(l).transpose() / input.cols();
            ws.bias_grad.at(l - 1) = ws.delta.at(l - 1).rowwise().mean();
        }

        return total_cost;
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
        for (size_t l = 1; l < depth(); l++) {
            const Eigen::MatrixXf& weight = get_weight(l);
            const Eigen::MatrixXf& bias = get_bias(l);

            file.write(reinterpret_cast<const char*>(weight.data()), weight.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(float));
        }
    }

    // gets the weight matrix connecting l-1 to l
    Eigen::MatrixXf& get_weight(size_t l) {
        if (l < 1 || l >= depth()) {
            throw std::out_of_range("get_weight: l is out of range" + std::to_string(l));
        }
        return weights.at(l - 1);
    }
    const Eigen::MatrixXf& get_weight(size_t l) const {
        if (l < 1 || l >= depth()) {
            throw std::out_of_range("get_weight (const): l is out of range");
        }
        return weights.at(l - 1);
    }

    // gets the biases of layer l
    Eigen::MatrixXf& get_bias(size_t l) {
        if (l < 1 || l >= depth()) {
            throw std::out_of_range("get_bias: l is out of range");
        }
        return biases.at(l - 1);
    }
    const Eigen::MatrixXf& get_bias(size_t l) const {
        if (l < 1 || l >= depth()) {
            throw std::out_of_range("get_bias (const): l is out of range");
        }
        return biases.at(l - 1);
    }

    // gets the activation function of layer l
    ActivationFunc get_activation_func(size_t l) const {
        if (l < 1 || l >= depth()) {
            throw std::out_of_range("get_activation_func: l is out of range" + std::to_string(l));
        }
        return activation_functions.at(l - 1);
    }

    FFNN() = delete;
private:
    FFNN(std::vector<size_t> network_shape, std::vector<ActivationFunc> activation_functions)
        : network_shape(std::move(network_shape)), activation_functions(std::move(activation_functions)) {}
};
