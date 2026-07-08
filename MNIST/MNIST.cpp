/*
clang++ -std=c++23 -O3 -march=native -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" ./MNIST/MNIST.cpp -L"$(brew --prefix libomp)/lib" -lomp -o test && ./test
*/

#include "../DataSet.hpp"
#include "../AdamOptimiser.hpp"
#include "../Trainer.hpp"
#include <string>
#include <filesystem>

constexpr int IMG_SIZE = 784;
constexpr int NUM_CLASSES = 10;

constexpr int EVAL_INTERVAL = 1000;

struct Settings {
    std::vector<size_t> ffnn_shape;
    std::vector<ActivationFunc> ffnn_funcs;
    CostType cost_type;
    RegularizationType reg_type;

    size_t num_steps;
    int seed;
    size_t batch_size;
    double lr;
    float noise;
    float chance_for_noise;
};

int main() {
    Settings settings{
        .ffnn_shape = {IMG_SIZE, 1024, 512, 255, 128, 64, 32, 16, NUM_CLASSES},
        .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax},
        .cost_type = CostType::categorical_cross_entropy,
        .reg_type = RegularizationType::none,
        .num_steps = 1000000,
        .seed = 1,
        .batch_size = 1000,
        .lr = 0.0005,
        .noise = 1,
        .chance_for_noise = 0.5
    };

    srand(settings.seed);

    DataSet test_dataset = DataSet::from_file("MNIST/bin/test.dat");
    DataSet train_dataset = DataSet::from_file("MNIST/bin/train.dat");

    FFNN ffnn = FFNN::from_random_he_scaling(settings.ffnn_shape, settings.ffnn_funcs);
    AdamOptimiser optimiser = AdamOptimiser::from_ffnn(ffnn, settings.cost_type, settings.lr, settings.reg_type);

    // randomly perturbs a fraction of the batch's columns towards white noise, see optimising.md
    auto add_noise = [&](Eigen::MatrixXf& inputs, Eigen::MatrixXf&) {
        for (Eigen::Index j = 0; j < inputs.cols(); j++) {
            if (static_cast<double>(rand()) / RAND_MAX < settings.chance_for_noise) {
                auto a = (Eigen::MatrixXf::Random(inputs.rows(), 1).array() + 1).matrix() * (settings.noise / 2.0f);
                inputs.col(j) += a;
            }
        }
    };

    TrainSettings train_settings{
        .num_steps = settings.num_steps,
        .batch_size = settings.batch_size,
        .eval_interval = EVAL_INTERVAL,
        .print_interval = 1,
        .seed = static_cast<unsigned int>(settings.seed),
        .checkpoint_path = "MNIST/models/best.dat"
    };

    TrainResult result = train(ffnn, optimiser, train_dataset, train_settings,
        [&](const FFNN& ffnn) { return nn_utils::argmax_accuracy(ffnn, test_dataset); },
        add_noise);

    std::cout << "\nbest test score " << result.best_score * 100 << "% at step " << result.best_step << "\n";
    std::cout << "final test score " << nn_utils::argmax_accuracy(ffnn, test_dataset) * 100 << "%\n";

    return 0;
}
