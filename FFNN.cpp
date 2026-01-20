// g++ -std=c++23 -g3 -ggdb -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o test.exe FFNN.cpp && test.exe
// g++ -std=c++23 -g3 -ggdb -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o test.exe FFNN.cpp && gdb test.exe

// g++ -std=c++23 -o test FFNN.cpp && test.exe

#include "Matrix.cpp"
#include <vector>
#include <iomanip>

enum ActivationFunc {
    linear,
    relu
};

enum CostType {
    quadratic
};

struct FFNN {
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    std::vector<int> layers;
    std::vector<ActivationFunc> activation_funcs;

    static FFNN from_random(std::vector<int> layers, std::vector<ActivationFunc> activation_funcs, float lo_weight = -1.0, float hi_weight = 1.0, float lo_bias = -1.0, float hi_bias = 1.0) {
        FFNN ffnn;

        ffnn.layers = layers;
        ffnn.activation_funcs = activation_funcs;

        if (layers.size() != activation_funcs.size() + 1) {
            std::cout << layers.size() << "  " << activation_funcs.size();
            std::cout << "activation funcs is not 1 smaller than layers!!!" << std::endl;
        }

        if (layers.size() <= 1) {
            std::cout << "input too small" << std::endl;
        }

        ffnn.weights.reserve(layers.size() - 1);
        ffnn.biases.reserve(layers.size() - 1);

        for (int l = 1; l < layers.size(); l++) {
            size_t width = static_cast<size_t>(layers.at(l - 1));
            size_t height = static_cast<size_t>(layers.at(l));

            ffnn.weights.push_back(Matrix::from_random(height, width, lo_weight, hi_weight));
            ffnn.biases.push_back(Matrix::from_random(height, 1, lo_bias, hi_bias));
        }

        return ffnn;
    }

    float activate(float x, ActivationFunc af) {
        switch (af) {
            case ActivationFunc::linear:
                return x;
            
            case ActivationFunc::relu:
                return x <= 0 ? 0.0 : x;
        }

        return 0.0;
    }

    Matrix activate(Matrix m, ActivationFunc af) {
        for (size_t i = 0; i < m.width * m.height; i++) {
            m.data[i] = activate(m.data[i], af);
        }
        return m;
    }

    float activate_der(float x, ActivationFunc af) {
        switch (af) {
            case ActivationFunc::linear:
                return 1.0;
            
            case ActivationFunc::relu:
                return x <= 0 ? 0.0 : 1.0;
        }
        return 0.0;
    }

    Matrix activate_der(Matrix m, ActivationFunc af) {
        for (size_t i = 0; i < m.width * m.height; i++) {
            m.data[i] = activate_der(m.data[i], af);
        }

        return m;
    }

    float cost_der(float a, float y, CostType type) {
        switch (type) {
        case CostType::quadratic:
            return a - y;
            break;
        default:
            break;
        }
        return 0.0;
    }

    Matrix cost_der(Matrix a, Matrix y, CostType type) {
        Matrix c(a.height, 1);
        for (size_t i = 0; i < a.height; i++) {
            c.at(i, 0) = cost_der(a.at(i, 0), y.at(i, 0), type);
        }
        return c;
    }

    float cost(const Matrix& a, const Matrix& y, CostType type) {
        switch (type) {
        case CostType::quadratic: {
            float sum = 0.0;
            for (size_t i = 0; i < a.height; i++) {
                float diff = a.at(i, 0) - y.at(i, 0);
                sum += diff * diff;
            }
            return sum / 2.0;
        }
        default:
            break;
        }
        return 0.0;
    }


    Matrix forward(const Matrix& input) {
        if (input.width != 1 || input.height != layers[0]) {
            std::cout << "forward input wrong" << std::endl;
        }

        Matrix cur_layer = activate(weights[0] * input + biases[0], activation_funcs[0]);
        for(size_t i = 1; i  < layers.size() - 1; i++) {
            cur_layer = activate(weights[i] * cur_layer + biases[i], activation_funcs[i]);
        }

        return cur_layer;
    }
    
    void backwards(const Matrix& input, const Matrix& expected_out, float lr = 0.005) {
        // forward and keep track of z = (wx + b)
        std::vector<Matrix> zs;
        zs.reserve(layers.size() - 1);

        Matrix z = weights[0] * input + biases[0];
        zs.push_back(z);
        Matrix a = activate(z, activation_funcs[0]);

        for (size_t i = 1; i < layers.size() - 1; i++) {
            z = weights[i] * a + biases[i];
            zs.push_back(z);
            a = activate(z, activation_funcs[i]);
        }

        // output layer delta
        Matrix sigma_prime = activate_der(zs.back(), activation_funcs.back());
        Matrix partial_derivatives = cost_der(a, expected_out, CostType::quadratic);
        Matrix cur_error = Matrix::hadamard_product(sigma_prime, partial_derivatives);

        // backprop through weight layers (0 .. weights.size()-1)
        for (int l = (int)weights.size() - 1; l >= 0; --l) {
            // compute prev activation (a_{l})
            Matrix prev_a = (l == 0) ? input : activate(zs[l - 1], activation_funcs[l - 1]);

            // update biases[l]
            for (size_t i = 0; i < biases[l].height; i++) {
                biases[l].at(i, 0) -= lr * cur_error.at(i, 0);
            }

            // update weights[l]
            for (size_t y = 0; y < weights[l].height; y++) {
                for (size_t x = 0; x < weights[l].width; x++) {
                    weights[l].at(y, x) -= lr * cur_error.at(y, 0) * prev_a.at(x, 0);
                }
            }

            // compute error for previous layer (skip when l == 0)
            if (l > 0) {
                Matrix sp = activate_der(zs[l - 1], activation_funcs[l - 1]);
                cur_error = Matrix::hadamard_product(weights[l].transposed() * cur_error, sp);
            }
        }
    }

    private:
    FFNN() {}
};

void train(std::vector<Matrix> inputs, std::vector<Matrix> outputs, FFNN& ffnn, size_t generations) {
    for (int gen = 0; gen < generations; gen++) {
        float total_cost = 0;
        for (int i = 0; i < inputs.size(); i++) {
            Matrix out = ffnn.forward(inputs[i]);
            total_cost += ffnn.cost(out, outputs[i], CostType::quadratic);

            ffnn.backwards(inputs[i], outputs[i]);
        }

        std::cout << "Finished generation " << gen << " || Cost = " << std::setprecision(4) << (total_cost / inputs.size()) << std::endl;
    }
}

int main() {
    FFNN net = FFNN::from_random(
        {3, 20, 3},
        {ActivationFunc::relu, ActivationFunc::relu},
        0.0, 1.0,
        0.0, 1.0
    );

    std::vector<Matrix> inputs;
    inputs.push_back(Matrix(3, 1, {1, 0, 0}));
    inputs.push_back(Matrix(3, 1, {0, 1, 0}));
    inputs.push_back(Matrix(3, 1, {0, 0, 1}));

    std::vector<Matrix> outputs;
    outputs.push_back(Matrix(3, 1, {1, 0, 0}));
    outputs.push_back(Matrix(3, 1, {1, 1, 0}));
    outputs.push_back(Matrix(3, 1, {1, 1, 1}));

    train(inputs, outputs, net, 10);
}
