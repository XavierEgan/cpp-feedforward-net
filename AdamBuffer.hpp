#pragma once
#include "Eigen/Dense"
#include "NN_Utils.hpp"

#include <vector>
#include <stdexcept>
#include <iostream>

struct AdamBuffer {
    std::vector<Eigen::MatrixXf> m_weights;
    std::vector<Eigen::MatrixXf> v_weights;
    std::vector<Eigen::MatrixXf> m_biases;
    std::vector<Eigen::MatrixXf> v_biases;
    int depth;

    AdamBuffer() = delete;
    AdamBuffer(int depth, std::vector<size_t> network_shape) : depth(depth) {
        m_weights.resize(depth - 1);
        v_weights.resize(depth - 1);
        m_biases.resize(depth - 1);
        v_biases.resize(depth - 1);

        for (int l = 1; l < depth; l++) {
            set_weight_m(l, Eigen::MatrixXf::Zero(network_shape.at(l), network_shape.at(l - 1)));
            set_weight_v(l, Eigen::MatrixXf::Zero(network_shape.at(l), network_shape.at(l - 1)));
            set_bias_m(l, Eigen::MatrixXf::Zero(network_shape.at(l), 1));
            set_bias_v(l, Eigen::MatrixXf::Zero(network_shape.at(l), 1));
        }
    }

    const Eigen::MatrixXf& get_weight_m(int l) const {
        nn_utils::check_layer_in_range(l, depth);
        return m_weights.at(l - 1);
    }
    const Eigen::MatrixXf& get_weight_v(int l) const {
        nn_utils::check_layer_in_range(l, depth);
        return v_weights.at(l - 1);
    }
    const Eigen::MatrixXf& get_bias_m(int l) const {
        nn_utils::check_layer_in_range(l, depth);
        return m_biases.at(l - 1);
    }
    const Eigen::MatrixXf& get_bias_v(int l) const {
        nn_utils::check_layer_in_range(l, depth);
        return v_biases.at(l - 1);
    }

    void set_weight_m(int l, const Eigen::MatrixXf& m) {
        nn_utils::check_layer_in_range(l, depth);
        m_weights.at(l - 1) = m;
    }
    void set_weight_v(int l, const Eigen::MatrixXf& v) {
        nn_utils::check_layer_in_range(l, depth);
        v_weights.at(l - 1) = v;
    }
    void set_bias_m(int l, const Eigen::MatrixXf& m) {
        nn_utils::check_layer_in_range(l, depth);
        m_biases.at(l - 1) = m;
    }
    void set_bias_v(int l, const Eigen::MatrixXf& v) {
        nn_utils::check_layer_in_range(l, depth);
        v_biases.at(l - 1) = v;
    }
};
