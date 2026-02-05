#pragma once
#include "Eigen/Dense"

enum ActivationFunc {
    linear,
    relu,
    relu_clipped,
    sigmoid,
    softmax
};

namespace nn_utils {
Eigen::MatrixXf activate(const Eigen::MatrixXf& z, ActivationFunc func) {
    switch (func) {
        case linear:
            return z;
        case relu:
            return z.cwiseMax(0.0f);
        case relu_clipped:
            return z.cwiseMax(0.0f).cwiseMin(1.0f);
        case sigmoid:
            return 1.0f / ((-z).array().exp() + 1);
        case softmax: {
            // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            // numerically stable softmax
            Eigen::MatrixXf shiftz = z.array().rowwise() - z.array().colwise().maxCoeff();
            Eigen::MatrixXf exps = shiftz.array().exp();
            return exps.array().rowwise() / exps.array().colwise().sum();
        }
        default:
            throw std::invalid_argument("activate: unknown activation function");
    }
}

Eigen::MatrixXf activate_der(const Eigen::MatrixXf& a, ActivationFunc func) {
    switch (func) {
        case linear:
            return Eigen::MatrixXf::Ones(a.rows(), a.cols());
        case relu:
            return a.unaryExpr([](float v) -> float { return v >= 0.0 ? 1.0 : 0.0; });
        case relu_clipped:
            return a.unaryExpr([](float v) -> float { return v >= 0.0 ? (v <= 1.0 ? 1.0 : 0.0 ) : 0.0; });
        case sigmoid:
            return a.unaryExpr([](float v) -> float { float s = 1.0f / (1.0f + std::exp(-v)); return s * (1.0f - s); });
        default:
            throw std::invalid_argument("activate: unknown activation function");
    }
}

}