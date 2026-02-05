#pragma once
#include "Eigen/Dense"
#include "NN_Utils.hpp"
#include <vector>

struct ForwardResult {
    ForwardResult() = delete;
    ForwardResult(int depth, std::vector<size_t> network_shape) : depth(depth) {
        zs.resize(depth - 1);
        as.resize(depth);

        for (int l = 1; l < depth; l++) {
            set_z(l, Eigen::MatrixXf::Zero(network_shape.at(l), 1));
            set_a(l, Eigen::MatrixXf::Zero(network_shape.at(l), 1));
        }
    }

    void set_z(int l, const Eigen::MatrixXf& z) {
        nn_utils::check_layer_in_range(l, depth);
        zs.at(l - 1) = z;
    }
    void set_a(int l, const Eigen::MatrixXf& a) {
        if (l != 0) {
            nn_utils::check_layer_in_range(l, depth);
        }
        as.at(l) = a;
    }

    const Eigen::MatrixXf& get_z(int l) const {
        nn_utils::check_layer_in_range(l, depth);
        return zs.at(l - 1);
    };

    const Eigen::MatrixXf& get_a(int l) const {
        // if a is zero then dont check range because its allowed to be zero
        if (l != 0) {
            nn_utils::check_layer_in_range(l, depth);
        }
        return as.at(l);
    };

private:
    int depth;
    std::vector<Eigen::MatrixXf> zs;
    std::vector<Eigen::MatrixXf> as;
};
