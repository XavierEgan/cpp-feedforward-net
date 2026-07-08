#pragma once
#include "Eigen/Dense"

#include <vector>
#include <stdexcept>

// scratch buffers for a forward pass, sized by network_shape and reused across calls
// buffers only reallocate when the batch size (number of columns) changes
struct ForwardWorkspace {
    std::vector<Eigen::MatrixXf> z;   // z.at(l - 1) holds layer l's pre-activation, l in 1..depth-1
    std::vector<Eigen::MatrixXf> a;   // a.at(l - 1) holds layer l's activation

    static ForwardWorkspace from_shape(const std::vector<size_t>& network_shape) {
        if (network_shape.size() < 2) {
            throw std::invalid_argument("from_shape: network_shape must have at least 2 layers");
        }

        return ForwardWorkspace(network_shape);
    }

    ForwardWorkspace() = delete;
private:
    explicit ForwardWorkspace(const std::vector<size_t>& network_shape) {
        z.resize(network_shape.size() - 1);
        a.resize(network_shape.size() - 1);
    }
};

// scratch buffers for a backward pass: a forward workspace plus per-layer error terms and gradients
// weight_grad/bias_grad are sized once up front since their shape never changes between calls
struct BackpropWorkspace {
    ForwardWorkspace fwd;
    std::vector<Eigen::MatrixXf> delta;           // delta.at(l - 1): error of layer l
    std::vector<Eigen::MatrixXf> activation_der;  // scratch for activate_der, same shape as delta
    std::vector<Eigen::MatrixXf> weight_grad;     // weight_grad.at(l - 1): shape[l] x shape[l - 1]
    std::vector<Eigen::MatrixXf> bias_grad;       // bias_grad.at(l - 1): shape[l] x 1

    static BackpropWorkspace from_shape(const std::vector<size_t>& network_shape) {
        if (network_shape.size() < 2) {
            throw std::invalid_argument("from_shape: network_shape must have at least 2 layers");
        }

        return BackpropWorkspace(network_shape);
    }

    BackpropWorkspace() = delete;
private:
    explicit BackpropWorkspace(const std::vector<size_t>& network_shape) : fwd(ForwardWorkspace::from_shape(network_shape)) {
        const size_t depth = network_shape.size();

        delta.resize(depth - 1);
        activation_der.resize(depth - 1);
        weight_grad.resize(depth - 1);
        bias_grad.resize(depth - 1);

        for (size_t l = 1; l < depth; l++) {
            weight_grad.at(l - 1) = Eigen::MatrixXf::Zero(network_shape.at(l), network_shape.at(l - 1));
            bias_grad.at(l - 1) = Eigen::MatrixXf::Zero(network_shape.at(l), 1);
        }
    }
};
