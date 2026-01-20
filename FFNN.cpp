// g++ -std=c++23 -g3 -ggdb -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o test.exe FFNN.cpp && test.exe
// g++ -std=c++23 -g3 -ggdb -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o test.exe FFNN.cpp && gdb test.exe

// g++ -std=c++23 -mavx -march=native -O3 -o test FFNN.cpp && test.exe

#include "Matrix.cpp"
#include <vector>
#include <iomanip>
#include <cmath>

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
    
    void backwards(const Matrix& input, const Matrix& expected_out, float lr = 0.00005) {
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
            // compute error for previous layer using CURRENT (pre-update) weights
            Matrix next_error;
            if (l > 0) {
                Matrix sp = activate_der(zs[l - 1], activation_funcs[l - 1]);
                next_error = Matrix::hadamard_product(weights[l].transposed() * cur_error, sp);
            }

            Matrix prev_a = (l == 0) ? input : activate(zs[l - 1], activation_funcs[l - 1]);

            for (size_t i = 0; i < biases[l].height; i++) {
                biases[l].at(i, 0) -= lr * cur_error.at(i, 0);
            }

            for (size_t y = 0; y < weights[l].height; y++) {
                for (size_t x = 0; x < weights[l].width; x++) {
                    weights[l].at(y, x) -= lr * cur_error.at(y, 0) * prev_a.at(x, 0);
                }
            }

            if (l > 0) cur_error = next_error;
        }
    }

    private:
    FFNN() {}
};

void train(std::vector<Matrix> inputs, std::vector<Matrix> outputs, FFNN& ffnn, size_t generations, float lr_start = 0.005, float lr_end = 0.005 ) {
    float prev_cost = std::numeric_limits<float>::max();

    for (int gen = 0; gen < generations; gen++) {
        // lerp between lr_start and lr_end
        float lr = lr_start + (lr_end - lr_start) * ((float)gen / (float)(generations - 1));

        float total_cost = 0;
        for (int i = 0; i < inputs.size(); i++) {
            Matrix out = ffnn.forward(inputs[i]);
            total_cost += ffnn.cost(out, outputs[i], CostType::quadratic);

            ffnn.backwards(inputs[i], outputs[i], lr);
        }

        std::cout << "Finished generation " << gen + 1 << " || Cost = " << std::fixed << std::setprecision(4) << (total_cost / inputs.size()) << " || Percentage improvement: " << ((prev_cost - (total_cost / inputs.size())) / prev_cost) * 100.0f << "%" << std::endl;

        prev_cost = total_cost / inputs.size();
    }



}

void test1() {
    FFNN net = FFNN::from_random(
        {3, 200, 3},
        {ActivationFunc::relu, ActivationFunc::linear},
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

    train(inputs, outputs, net, 10000, 0.01, 0.0001);
}

void test2() {
    FFNN net = FFNN::from_random(
        {2, 4, 1},
        {ActivationFunc::relu, ActivationFunc::linear},
        -1.0, 1.0,
        -1.0, 1.0
    );

    std::vector<Matrix> inputs;
    std::vector<Matrix> outputs;

    for (int i = 0; i < 1000; i++) {
        float x1 = (((float)rand()) / ((float)RAND_MAX)) * 10.0 - 5.0;
        float x2 = (((float)rand()) / ((float)RAND_MAX)) * 10.0 - 5.0;

        inputs.push_back(Matrix(2, 1, {x1, x2}));
        outputs.push_back(Matrix(1, 1, { (float)(x1 + x2) / 2.0f }));
    }

    train(inputs, outputs, net, 500, 0.01, 0.0001);
}

void test3() {
    FFNN net = FFNN::from_random(
        {2, 4, 4, 1},
        {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::linear},
        -1.0, 1.0,
        -1.0, 1.0
    );

    std::vector<Matrix> inputs;
    std::vector<Matrix> outputs;

    for (int i = 0; i < 1000; i++) {
        float x1 = (((float)rand()) / ((float)RAND_MAX)) * 10.0 - 5.0;
        float x2 = (((float)rand()) / ((float)RAND_MAX)) * 10.0 - 5.0;

        inputs.push_back(Matrix(2, 1, {x1, x2}));
        outputs.push_back(Matrix(1, 1, { sin((float)(x1 + x2)) }));
    }
    train(inputs, outputs, net, 5000, 0.01, 0.0005);
}

int main() {
    test3();
}
