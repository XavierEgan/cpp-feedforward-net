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

static void check_layer_in_range(int l, int depth) {
    if (l < 1 || l >= depth) {
        throw std::out_of_range("get_z: l is out of range: " + std::to_string(l));
    }
}

static void check_matrix_shape(const Eigen::MatrixXf& m, int rows, int cols, const std::string& name) {
    if (m.rows() != rows || m.cols() != cols) {
        throw std::invalid_argument(name + " has incorrect shape. Expected (" + std::to_string(rows) + ", " + std::to_string(cols) + "), got (" + std::to_string(m.rows()) + ", " + std::to_string(m.cols()) + ")");
    }
}

template<typename T>
static std::string get_matrix_shape_str(const T& m) {
    return "(" + std::to_string(m.rows()) + ", " + std::to_string(m.cols()) + ")";
}

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
        check_layer_in_range(l, depth);
        weight_gradients.at(l - 1) = grad;
    }
    void set_bias_gradient(int l, const Eigen::MatrixXf& grad) {
        check_layer_in_range(l, depth);
        bias_gradients.at(l - 1) = grad;
    }

    const Eigen::MatrixXf& get_weight_gradient(int l) const {
        check_layer_in_range(l, depth);
        return weight_gradients.at(l - 1);
    }
    const Eigen::MatrixXf& get_bias_gradient(int l) const {
        check_layer_in_range(l, depth);
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
        check_layer_in_range(l, depth);
        zs.at(l - 1) = z;
    }
    void set_a(int l, const Eigen::MatrixXf& a) {
        if (l != 0) {
            check_layer_in_range(l, depth);
        }
        as.at(l) = a;
    }

    const Eigen::MatrixXf& get_z(int l) const {
        check_layer_in_range(l, depth);
        return zs.at(l - 1);
    };

    const Eigen::MatrixXf& get_a(int l) const {
        // if a is zero then dont check range because its allowed to be zero
        if (l != 0) {
            check_layer_in_range(l, depth);
        }
        return as.at(l);
    };

private:
    int depth;
    std::vector<Eigen::MatrixXf> zs;
    std::vector<Eigen::MatrixXf> as;
};

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
        check_layer_in_range(l, depth);
        return m_weights.at(l - 1);
    }
    const Eigen::MatrixXf& get_weight_v(int l) const {
        check_layer_in_range(l, depth);
        return v_weights.at(l - 1);
    }
    const Eigen::MatrixXf& get_bias_m(int l) const {
        check_layer_in_range(l, depth);
        return m_biases.at(l - 1);
    }
    const Eigen::MatrixXf& get_bias_v(int l) const {
        check_layer_in_range(l, depth);
        return v_biases.at(l - 1);
    }

    void set_weight_m(int l, const Eigen::MatrixXf& m) {
        check_layer_in_range(l, depth);
        m_weights.at(l - 1) = m;
    }
    void set_weight_v(int l, const Eigen::MatrixXf& v) {
        check_layer_in_range(l, depth);
        v_weights.at(l - 1) = v;
    }
    void set_bias_m(int l, const Eigen::MatrixXf& m) {
        check_layer_in_range(l, depth);
        m_biases.at(l - 1) = m;
    }
    void set_bias_v(int l, const Eigen::MatrixXf& v) {
        check_layer_in_range(l, depth);
        v_biases.at(l - 1) = v;
    }
};

struct FFNN {
    size_t depth;
    std::vector<size_t> network_shape;
    std::vector<ActivationFunc> activation_functions;
    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::MatrixXf> biases;
    CostType cost_type;

    ForwardResult forward_result;
    BackpropResult backprop_result;
    AdamBuffer adam_buffer;

    static constexpr float kProbEps = 1e-7f;

    static FFNN from_random(const std::vector<size_t>& network_shape, const std::vector<ActivationFunc>& activation_funcs, CostType cost_type = CostType::quadratic) {
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
        AdamBuffer adam_buffer = AdamBuffer(depth, network_shape);
        
        FFNN ffnn(forward_result, backprop_result, adam_buffer);

        ffnn.depth = depth;
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

    static Eigen::MatrixXf activate(const Eigen::MatrixXf& x, ActivationFunc func) {
        switch (func) {
            case linear:
                return x;
            case relu:
                return x.cwiseMax(0.0f);
            case relu_clipped:
                return x.cwiseMax(0.0f).cwiseMin(1.0f);
            case sigmoid:
                return 1.0f / ((-x).array().exp() + 1);
            default:
                throw std::invalid_argument("activate: unknown activation function");
        }
    }

    static Eigen::MatrixXf activate_der(const Eigen::MatrixXf& x, ActivationFunc func) {
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

    static Eigen::MatrixXf cost_derivative(const Eigen::MatrixXf& a, const Eigen::MatrixXf& y, CostType cost_type) {
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
                return ((a - y).cwiseSquare() * 0.5).sum() / (a.rows() * a.cols());
            }
            case binary_cross_entropy: {
                Eigen::ArrayXXf ac = a.array().min(1.0f - kProbEps).max(kProbEps);
                return (-(y.array() * ac.log()) - (1.0f - y.array()) * (1.0f - ac).log()).sum() / (a.rows() * a.cols());
            }
            default:
                throw std::invalid_argument("cost: unknown cost type");
        }
    }

    static void get_batch(const std::vector<Eigen::MatrixXf>& inputs, const std::vector<Eigen::MatrixXf>& targets, Eigen::MatrixXf& minibatch, Eigen::MatrixXf& minibatch_targets, int batch_size = -1) {
        if (inputs.size() != targets.size()) {
            throw std::invalid_argument("get_batch: inputs and targets must be the same size");
        }
        if (inputs.size() == 0) {
            throw std::invalid_argument("get_batch: inputs and targets cannot be empty");
        }

        const int n =  inputs.size();

        if (batch_size <= 0) batch_size = n;
        if (batch_size > n) batch_size = n;

        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        
        static thread_local std::mt19937 rng(std::random_device{}());

        if (batch_size < n) {
            std::shuffle(indices.begin(), indices.end(), rng);
            indices.resize(batch_size);
        }

        minibatch.resize(inputs.at(0).rows(), batch_size);
        minibatch_targets.resize(targets.at(0).rows(), batch_size);

        for (int b = 0; b < batch_size; b++) {
            // std::cout << "here" << std::endl;

            // std::cout << "Minibatch size: " << get_matrix_shape_str(minibatch) << std::endl;

            // std::cout << "LHS Shape: " << get_matrix_shape_str(minibatch.col(b)) << ", RHS Shape: " << get_matrix_shape_str(inputs.at(indices.at(b)).col(0)) << std::endl;
            
            minibatch.col(b) = inputs.at(indices.at(b));
            minibatch_targets.col(b) = targets.at(indices.at(b));
        }
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        Eigen::MatrixXf prev_a = input;
        for (size_t l = 1; l < depth; l++) {
            
            Eigen::MatrixXf z = (get_weight(l) * prev_a).colwise() + get_bias(l).col(0);
            prev_a = activate(z, get_activation_func(l));
        }
        return prev_a;
    }

    void forward(ForwardResult& forward_result) {
        for (size_t l = 1; l < depth; l++) {
            Eigen::MatrixXf z = (get_weight(l) * forward_result.get_a(l - 1)).colwise() + get_bias(l).col(0);
            forward_result.set_z(l, z);
            forward_result.set_a(l, activate(z, get_activation_func(l)));
        }
    }

    void backpropogate(const Eigen::MatrixXf& input, const Eigen::MatrixXf& target) {
        forward_result.set_a(0, input);
        forward(forward_result);

        backprop_result.get_cost() = cost(forward_result.get_a(depth - 1), target, cost_type);

        // get errors in the output layer
        Eigen::MatrixXf output_error = cost_derivative(forward_result.get_a(depth - 1), target, cost_type).cwiseProduct(activate_der(forward_result.get_z(depth - 1), get_activation_func(depth - 1)));
        Eigen::MatrixXf prev_error = output_error;

        backprop_result.set_weight_gradient(depth - 1, output_error * forward_result.get_a(depth - 2).transpose() / input.cols());
        backprop_result.set_bias_gradient(depth - 1, output_error.rowwise().mean());
        
        // get error in all future layers
        for (size_t l = depth - 2; l > 0; l--) {
            // weight connecting this layer to the next
            // the error in the next layer
            // and the derivative of the activation function of this layer
            Eigen::MatrixXf this_error = (get_weight(l + 1).transpose() * prev_error).cwiseProduct(activate_der(forward_result.get_z(l), get_activation_func(l)));
            
            backprop_result.set_weight_gradient(l, this_error * forward_result.get_a(l - 1).transpose() / input.cols());
            backprop_result.set_bias_gradient(l, this_error.rowwise().mean());
            
            prev_error = this_error;
        }
    }

    float gradient_descent(const Eigen::MatrixXf& minibatch, const Eigen::MatrixXf& minibatch_targets, double lr = 0.005) {
        const int minibatch_size = minibatch.cols();

        if (minibatch_size <= 0) {
            throw std::invalid_argument("gradient_descent: minibatch size must be non-zero");
        }

        // backprop to get gradients
        backpropogate(minibatch, minibatch_targets);

        // update weights and biases
        for (size_t l = 1; l < depth; l++) {
            get_weight(l) -= lr * backprop_result.get_weight_gradient(l);
            get_bias(l) -= lr * backprop_result.get_bias_gradient(l);
        }

        return backprop_result.get_cost();
    }

    float adam(const Eigen::MatrixXf& minibatch, const Eigen::MatrixXf& minibatch_targets, int t, double lr = 0.005, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) {
        const int minibatch_size = minibatch.cols();

        if (minibatch_size <= 0) {
            throw std::invalid_argument("gradient_descent: minibatch size must be non-zero");
        }

        // backprop to get gradients
        backpropogate(minibatch, minibatch_targets);
        
        // update weights and biases
        for (size_t l = 1; l < depth; l++) {
            adam_buffer.set_weight_m(l, beta1 * adam_buffer.get_weight_m(l) + (1.0 - beta1) * backprop_result.get_weight_gradient(l));
            adam_buffer.set_weight_v(l, beta2 * adam_buffer.get_weight_v(l) + (1.0 - beta2) * backprop_result.get_weight_gradient(l).cwiseSquare());

            Eigen::MatrixXf m_hat = adam_buffer.get_weight_m(l) / (1.0 - std::pow(beta1, t));
            Eigen::MatrixXf v_hat = adam_buffer.get_weight_v(l) / (1.0 - std::pow(beta2, t));

            get_weight(l) -= lr * m_hat.cwiseQuotient((v_hat.array().sqrt() + epsilon).matrix());

            adam_buffer.set_bias_m(l, beta1 * adam_buffer.get_bias_m(l) + (1.0 - beta1) * backprop_result.get_bias_gradient(l));
            adam_buffer.set_bias_v(l, beta2 * adam_buffer.get_bias_v(l) + (1.0 - beta2) * backprop_result.get_bias_gradient(l).cwiseSquare());

            Eigen::MatrixXf m_hat_b = adam_buffer.get_bias_m(l) / (1.0 - std::pow(beta1, t));
            Eigen::MatrixXf v_hat_b = adam_buffer.get_bias_v(l) / (1.0 - std::pow(beta2, t));

            get_bias(l) -= lr * m_hat_b.cwiseQuotient((v_hat_b.array().sqrt() + epsilon).matrix());
        }

        return backprop_result.get_cost();
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
    FFNN(ForwardResult forward_result, BackpropResult backprop_result, AdamBuffer adam_buffer) : forward_result(forward_result), backprop_result(backprop_result), adam_buffer(adam_buffer) {}
};

void train(FFNN& ffnn, const std::vector<Eigen::MatrixXf>& inputs, const std::vector<Eigen::MatrixXf>& targets, int generations, int batch_size = -1, double lr = 0.005, double decay = .999) {
    for (int gen = 0; gen < generations; gen++) {
        Eigen::MatrixXf minibatch;
        Eigen::MatrixXf minibatch_targets;
        FFNN::get_batch(inputs, targets, minibatch, minibatch_targets, batch_size);
        float avg_cost = ffnn.gradient_descent(minibatch, minibatch_targets, lr);

        std::cout << "Generation " << gen << "  Avg Cost: " << avg_cost << "  lr: " << lr << std::endl;

        lr *= decay;
    }
}

struct DecayOnPlateauScheduler {
    double learning_rate;
    double min_learning_rate;
    double decay;
    int patience;
    float best_metric;
    int epochs_no_improve;

    DecayOnPlateauScheduler(double start_lr = 0.01, double min_lr = 0.001, double decay = 0.5, int patience = 5)
        : learning_rate(start_lr), min_learning_rate(min_lr), decay(decay), patience(patience), best_metric(std::numeric_limits<float>::max()), epochs_no_improve(0) {
            std::cout << "DecayOnPlateauScheduler, Min learning steps : " << log(min_learning_rate / learning_rate) / log(decay) * patience << std::endl;
        }

    bool step(float current_metric) {
        if (current_metric < best_metric) {
            best_metric = current_metric;
            epochs_no_improve = 0;
        } else {
            epochs_no_improve++;
            if (epochs_no_improve >= patience) {
                learning_rate *= decay;
                if (learning_rate < min_learning_rate) {
                    // were dont with training
                    return true;
                }
                epochs_no_improve = 0;
            }
        }
        return false;
    }
};

// void test1() {
//     FFNN ffnn = FFNN::from_random(
//         {2, 128, 128, 2},
//         {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::linear}
//     );

//     std::vector<Eigen::MatrixXf> inputs;
//     std::vector<Eigen::MatrixXf> targets;

//     for (int i = 0; i < 100; i++) {
//         inputs.push_back(Eigen::MatrixXf::Random(2, 1));
//         targets.push_back(Eigen::MatrixXf::Random(2, 1));
//     }

//     train(ffnn, inputs, targets, 10000, 10, 0.05);
// }

// void test2() {
//     FFNN ffnn = FFNN::from_random(
//     {16, 128, 512, 512, 128, 16},
//     {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::linear}
//     );

//     std::vector<Eigen::MatrixXf> inputs;
//     std::vector<Eigen::MatrixXf> targets;

//     for (int i = 0; i < 1000; i++) {
//         inputs.push_back(Eigen::MatrixXf::Random(16, 1).array() * 100);
//         targets.push_back(Eigen::MatrixXf::Random(16, 1).array() * 100);
//     }

//     train(ffnn, inputs, targets, 500, -1, 0.0001, .999);
// }

// int main() {
//     test2();
// }