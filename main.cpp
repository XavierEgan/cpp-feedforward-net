#include "GradientDescentOptimiser.hpp"
#include "AdamOptimiser.hpp"


int main() {
    std::vector<size_t> layer_sizes = {128, 256, 128};
    std::vector<ActivationFunc> activation_funcs = {ActivationFunc::relu, ActivationFunc::relu};

    FFNN ffnn_gd = FFNN::from_random_he_scaling(layer_sizes, activation_funcs);
    FFNN ffnn_adam = FFNN::from_random_he_scaling(layer_sizes, activation_funcs);

    GradientDescentOptimiser gd_optimiser(ffnn_gd, CostType::quadratic, 0.0005);
    AdamOptimiser adam_optimiser(ffnn_adam, CostType::quadratic);

    Eigen::MatrixXf input = (Eigen::MatrixXf::Random(128, 128) + Eigen::MatrixXf::Constant(128, 128, 1.0f)) / 2.0f;
    Eigen::MatrixXf target = (Eigen::MatrixXf::Random(128, 128) + Eigen::MatrixXf::Constant(128, 128, 1.0f)) / 2.0f;

    for (int i = 0; i < 10000; i++) {
        float gd_cost = gd_optimiser.step(input, target);
        float adam_cost = adam_optimiser.step(input, target);

        std::cout << "GD Cost: " << gd_cost << ", Adam Cost: " << adam_cost << std::endl;
    }

    return 0;
}