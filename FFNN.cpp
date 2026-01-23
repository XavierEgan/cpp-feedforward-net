// g++ -std=c++23 -g3 -ggdb -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o test.exe FFNN.cpp && test.exe
// g++ -std=c++23 -g3 -ggdb -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o test.exe FFNN.cpp && gdb test.exe

// g++ -std=c++23 -march=native -O3 -o test FFNN.cpp && test.exe
// g++ -std=c++23 -march=native -DNDEBUG -O3 -o test FFNN.cpp && ./test

#include "Eigen/Dense"
#include <vector>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>


enum ActivationFunc {
    linear,
    relu,
    relu_clipped,
    sigmoid
};

enum CostType {
    quadratic,
    binary_cross_entropy
};

struct BackpropResult {
    std::vector<Eigen::MatrixXf> weight_gradients;
    std::vector<Eigen::MatrixXf> bias_gradients;
    float cost;
};

struct FFNN {
    size_t depth;
    std::vector<size_t> network_shape;
    std::vector<ActivationFunc> activation_functions;
    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::MatrixXf> biases;
    CostType cost_type;

    static constexpr float kProbEps = 1e-7f;

    static FFNN from_random(std::vector<size_t> network_shape, std::vector<ActivationFunc> activation_funcs, CostType cost_type = CostType::quadratic) {
        FFNN ffnn;

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

        ffnn.depth = network_shape.size();
        ffnn.activation_functions = activation_funcs;
        ffnn.network_shape = network_shape;
        ffnn.cost_type = cost_type;

        // weights = output x input
        // He-style scaling helps keep activations/gradients stable for ReLU nets.
        for (size_t i = 1; i < ffnn.depth; i++) {
            const float fan_in = static_cast<float>(network_shape.at(i - 1));
            const float scale = std::sqrt(2.0f / fan_in);
            ffnn.weights.push_back(Eigen::MatrixXf::Random(network_shape.at(i), network_shape.at(i - 1)) * scale);
            ffnn.biases.push_back(Eigen::MatrixXf::Zero(network_shape.at(i), 1));
        }

        return ffnn;
    }

    Eigen::MatrixXf activate(const Eigen::MatrixXf& x, ActivationFunc func) {
        switch (func) {
            case linear:
                return x;
            case relu:
                return x.unaryExpr([](float v) -> float { return std::fmax(0.0f, v); });
            case relu_clipped:
                return x.unaryExpr([](float v) -> float { return std::fmax(0.0f, (std::fmin(1.0f, v))); });
            case sigmoid:
                return x.unaryExpr([](float v) -> float { return 1.0f / (1.0f + std::exp(-v)); });
            default:
                throw std::invalid_argument("activate: unknown activation function");
        }
    }

    Eigen::MatrixXf activate_der(const Eigen::MatrixXf& x, ActivationFunc func) {
        switch (func) {
            case linear:
                return x.unaryExpr([](float v) -> float { return 1.0; });
            case relu:
                return x.unaryExpr([](float v) -> float { return v >= 0.0 ? 1.0 : 0.0; });
            case relu_clipped:
                return x.unaryExpr([](float v) -> float { return v >= 0.0 ? (v <= 1.0 ? 1.0 : 0.0 ) : 0.0; });
            case sigmoid:
                return x.unaryExpr([](float v) -> float { float s = 1.0f / (1.0f + std::exp(-v)); return s * (1.0f - s); });
            default:
                throw std::invalid_argument("activate: unknown activation function");
        }
    }

    Eigen::MatrixXf cost_derivative(const Eigen::MatrixXf& a, const Eigen::MatrixXf& y, CostType cost_type) {
        switch (cost_type) {
            case quadratic:
                return a - y;
            case binary_cross_entropy: {
                 Eigen::ArrayXXf ac = a.array().min(1.0f - kProbEps).max(kProbEps);
                 return (-(y.array() / ac) + (1.0f - y.array()) / (1.0f - ac)).matrix();
            }
            default:
                throw std::invalid_argument("cost_derivative: unknown cost type");
        }
    }

    static float cost(const Eigen::MatrixXf& a, const Eigen::MatrixXf& y, CostType cost_type) {
        //std::cout << a << std::endl << std::endl;
        switch (cost_type) {
            case quadratic: {
                return ((a - y).cwiseSquare() * 0.5).sum() / a.rows();
            }
            case binary_cross_entropy: {
                Eigen::ArrayXXf ac = a.array().min(1.0f - kProbEps).max(kProbEps);
                return (-(y.array() * ac.log()) - (1.0f - y.array()) * (1.0f - ac).log()).sum() / a.rows();
            }
            default:
                throw std::invalid_argument("cost: unknown cost type");
        }
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        Eigen::MatrixXf prev_a = input;
        for (int l = 1; l < depth; l++) {
            Eigen::MatrixXf z = get_weight(l) * prev_a + get_bias(l);
            prev_a = activate(z, get_activation_func(l));
        }
        return prev_a;
    }

    void backpropogate(const Eigen::MatrixXf& input, const Eigen::MatrixXf& target, BackpropResult& result) {
        std::vector<Eigen::MatrixXf> as;
        std::vector<Eigen::MatrixXf> zs;

        as.reserve(depth);
        zs.reserve(depth - 1);

        as.push_back(input);

        Eigen::MatrixXf prev_a = input;
        for (int l = 1; l < depth; l++) {
            Eigen::MatrixXf z = get_weight(l) * prev_a + get_bias(l);
            zs.push_back(z);
            
            prev_a = activate(z, get_activation_func(l));
            as.push_back(prev_a);
        }
        // std::cout << a << std::endl << std::endl;

        auto get_z = [this, &zs](int l) -> const Eigen::MatrixXf& {
            if (l < 1 || l >= depth) {
                throw std::out_of_range("get_z: l is out of range " + std::to_string(l));
            }
            return zs.at(l - 1);
        };

        auto get_a = [this, &as](int l) -> const Eigen::MatrixXf& {
            if (l < 0 || l >= depth) {
                throw std::out_of_range("get_a: l is out of range " + std::to_string(l));
            }
            return as.at(l);
        };

        result.weight_gradients.clear();
        result.bias_gradients.clear();
        result.weight_gradients.reserve(depth - 1);
        result.bias_gradients.reserve(depth - 1);
        result.cost = cost(get_a(depth - 1), target, cost_type);

        // get errors in the output layer
        Eigen::MatrixXf output_error = cost_derivative(prev_a, target, cost_type).cwiseProduct(activate_der(get_z(depth - 1), get_activation_func(depth - 1)));
        Eigen::MatrixXf prev_error = output_error;

        result.weight_gradients.push_back(output_error * get_a(depth - 2).transpose());
        result.bias_gradients.push_back(output_error);
        // get error in all future layers
        for (int l = depth - 2; l > 0; l--) {
            // weight connecting this layer to the next
            // the error in the next layer
            // and the derivative of the activation function of this layer
            Eigen::MatrixXf this_error = (get_weight(l + 1).transpose() * prev_error).cwiseProduct(activate_der(get_z(l), get_activation_func(l)));
            
            result.weight_gradients.push_back(this_error * get_a(l - 1).transpose());
            result.bias_gradients.push_back(this_error);
            
            prev_error = this_error;
        }

        std::reverse(result.weight_gradients.begin(), result.weight_gradients.end());
        std::reverse(result.bias_gradients.begin(), result.bias_gradients.end());
    }

    float gradient_descent(const std::vector<Eigen::MatrixXf>& inputs, const std::vector<Eigen::MatrixXf>& targets, int batch_size = -1, float lr = 0.005f) {
        if (inputs.empty()) {
            throw std::invalid_argument("gradient_descent: inputs must be non-empty");
        }
        if (inputs.size() != targets.size()) {
            throw std::invalid_argument("gradient_descent: inputs and targets must be the same size");
        }

        const int n =  inputs.size();

        if (batch_size <= 0) batch_size = n;

        float total_cost = 0.0;
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        
        static thread_local std::mt19937 rng(std::random_device{}());

        if (batch_size < n) {
            std::shuffle(indices.begin(), indices.end(), rng);
            indices.resize(batch_size);
        }

        BackpropResult bp_result {{}, {}, 0.0f};
        std::vector<Eigen::MatrixXf> total_weight_gradient;
        std::vector<Eigen::MatrixXf> total_bias_gradient;

        for (int i = 0; i < depth - 1; i++) {
            total_weight_gradient.push_back(Eigen::MatrixXf::Zero(weights.at(i).rows(), weights.at(i).cols()));
            total_bias_gradient.push_back(Eigen::MatrixXf::Zero(biases.at(i).rows(), biases.at(i).cols()));
        }

        for (int idx : indices) {
            backpropogate(inputs.at(idx), targets.at(idx), bp_result);
            total_cost += bp_result.cost;
            for (int i = 0; i < depth - 1; i++) {
                total_weight_gradient.at(i) += bp_result.weight_gradients.at(i);
                total_bias_gradient.at(i) += bp_result.bias_gradients.at(i);
            }
        }

        // update weights and biases
        for (int l = 1; l < depth; l++) {
            get_weight(l) -= (lr / batch_size) * total_weight_gradient.at(l - 1);
            get_bias(l) -= (lr / batch_size) * total_bias_gradient.at(l - 1);
        }

        return total_cost / batch_size;
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
};

void train(FFNN& ffnn, const std::vector<Eigen::MatrixXf>& inputs, const std::vector<Eigen::MatrixXf>& targets, int generations, int batch_size = -1, double lr = 0.005f, double decay = .999) {
    for (int gen = 0; gen < generations; gen++) {
        float avg_cost = ffnn.gradient_descent(inputs, targets, batch_size, lr);

        std::cout << "Generation " << gen << "  Avg Cost: " << avg_cost << "  lr: " << lr << std::endl;

        lr *= decay;
    }
}

void test1() {
    FFNN ffnn = FFNN::from_random(
        {2, 128, 128, 2},
        {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::linear}
    );

    std::vector<Eigen::MatrixXf> inputs;
    std::vector<Eigen::MatrixXf> targets;

    for (int i = 0; i < 100; i++) {
        inputs.push_back(Eigen::MatrixXf::Random(2, 1));
        targets.push_back(Eigen::MatrixXf::Random(2, 1));
    }

    train(ffnn, inputs, targets, 10000, 10, 0.05);
}

void test2() {
    FFNN ffnn = FFNN::from_random(
    {16, 128, 512, 512, 128, 16},
    {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::linear}
    );

    std::vector<Eigen::MatrixXf> inputs;
    std::vector<Eigen::MatrixXf> targets;

    for (int i = 0; i < 1000; i++) {
        inputs.push_back(Eigen::MatrixXf::Random(16, 1).array() * 100);
        targets.push_back(Eigen::MatrixXf::Random(16, 1).array() * 100);
    }

    train(ffnn, inputs, targets, 500, -1, 0.0001, .999);
}

// int main() {
//     test2();

//     return 0;
// }
