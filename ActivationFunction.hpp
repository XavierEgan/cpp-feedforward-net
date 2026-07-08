#pragma once
#include "Eigen/Dense"

#include <cstdint>
#include <stdexcept>

enum class ActivationFunc : uint32_t {
    linear,
    relu,
    relu_clipped,
    tan_h,
    sigmoid,
    softmax
};

namespace nn_utils {

inline Eigen::MatrixXf activate(const Eigen::MatrixXf& z, ActivationFunc func) {
    switch (func) {
        case ActivationFunc::linear:
            return z;
        case ActivationFunc::relu:
            return z.cwiseMax(0.0f);
        case ActivationFunc::relu_clipped:
            return z.cwiseMax(0.0f).cwiseMin(1.0f);
        case ActivationFunc::sigmoid:
            return 1.0f / ((-z).array().exp() + 1);
        case ActivationFunc::softmax: {
            // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            // numerically stable softmax
            Eigen::MatrixXf shiftz = z.array().rowwise() - z.array().colwise().maxCoeff();
            Eigen::MatrixXf exps = shiftz.array().exp();
            return exps.array().rowwise() / exps.array().colwise().sum();
        }
        case ActivationFunc::tan_h:
            return z.array().tanh().matrix();
        default:
            throw std::invalid_argument("activate: unknown activation function");
    }
}

// writes the activation into out so the hot path never allocates on the caller's side
inline void activate(const Eigen::MatrixXf& z, ActivationFunc func, Eigen::MatrixXf& out) {
    out = activate(z, func);
}

inline Eigen::MatrixXf activate_der(const Eigen::MatrixXf& z, ActivationFunc func) {
    switch (func) {
        case ActivationFunc::linear:
            return Eigen::MatrixXf::Ones(z.rows(), z.cols());
        case ActivationFunc::relu:
            return z.unaryExpr([](float v) -> float { return v >= 0.0 ? 1.0 : 0.0; });
        case ActivationFunc::relu_clipped:
            return z.unaryExpr([](float v) -> float { return v >= 0.0 ? (v <= 1.0 ? 1.0 : 0.0 ) : 0.0; });
        case ActivationFunc::sigmoid:
            return z.unaryExpr([](float v) -> float { float s = 1.0f / (1.0f + std::exp(-v)); return s * (1.0f - s); });
        case ActivationFunc::tan_h:
            return Eigen::MatrixXf::Ones(z.rows(), z.cols()) - z.array().tanh().square().matrix();
        default:
            throw std::invalid_argument("activate_der: unknown activation function");
    }
}

inline void activate_der(const Eigen::MatrixXf& z, ActivationFunc func, Eigen::MatrixXf& out) {
    out = activate_der(z, func);
}

}