/*
clang++ -std=c++23 -O3 -march=native -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" ./MNIST/MNIST.cpp -L"$(brew --prefix libomp)/lib" -lomp -o test && ./test
*/

#include "../DataSet.hpp"
#include "../AdamOptimiser.hpp"
#include <string>
#include <filesystem>

constexpr int IMG_SIZE = 784;
constexpr int NUM_CLASSES = 10;
constexpr int TRAIN_SIZE = 60000;
constexpr int TEST_SIZE = 10000;

constexpr int DEBUG_PRINT_INTERVAL = 100;
constexpr int EVAL_PRINT_INTERVAL = 1000;

struct Settings {
    std::vector<size_t> ffnn_shape;
    std::vector<ActivationFunc> ffnn_funcs;
    CostType cost_type;
    RegularizationType reg_type;

    size_t num_epochs;
    int seed;
    size_t batch_size;
    double lr;
    float noise;
    float chance_for_noise;
};

float eval_on_test(const DataSet& test_data, const FFNN& ffnn, bool quiet = true) {
    int total_right = 0;

    // get all predictions in one forward pass
    const auto predictions = ffnn.forward(test_data.inputs);

    for (Eigen::Index i = 0; i < test_data.inputs.cols(); i++) {
        // answer
        Eigen::Index answer = 0, answer_col = 0;
        test_data.labels.col(i).maxCoeff(&answer, &answer_col);

        // prediction
        Eigen::Index prediction = 0, prediction_col = 0;
        predictions.col(i).maxCoeff(&prediction, &prediction_col);

        if (answer == prediction) total_right ++;
    }

    if (!quiet)
        std::cout << "Total test correct: " << total_right << " Percentage right: " << (static_cast<float>(total_right) / static_cast<float>(test_data.size())) * 100.0f << "%" << std::endl;

    return static_cast<float>(total_right) / static_cast<float>(test_data.size());
}

int main() {
    Settings settings{
        .ffnn_shape = {IMG_SIZE, 1024, 512, 255, 128, 64, 32, 16, NUM_CLASSES},
        .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax},
        .cost_type = CostType::categorical_cross_entropy,
        .reg_type = RegularizationType::none,
        .num_epochs = 1000000,
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
    Batcher batcher = Batcher::from_dataset(train_dataset, settings.seed);
    Eigen::MatrixXf inputs;
    Eigen::MatrixXf targets;

    float best_test_score = std::numeric_limits<float>::min();
    int best_test_epoch = 0;

    for (auto epoch{0uz}; epoch < settings.num_epochs; epoch++) {
        batcher.next_batch(settings.batch_size, inputs, targets);

        for (Eigen::Index j = 0; j < inputs.cols(); j++) {
            if (static_cast<double>(rand()) / RAND_MAX < settings.chance_for_noise) {
                auto a = (Eigen::MatrixXf::Random(inputs.rows(), 1).array() + 1).matrix() * (settings.noise / 2.0f);
                inputs.col(j) += a;
            }
        }

        // if (static_cast<double>(rand()) / RAND_MAX < settings.chance_for_noise)
        //     inputs += (Eigen::MatrixXf::Random(inputs.rows(), inputs.cols()).array() + 1).matrix() * settings.noise;

        float cost = optimiser.step(inputs, targets);

        if (epoch % EVAL_PRINT_INTERVAL == 0) {
            float test_score = eval_on_test(test_dataset, ffnn, false);
            if (test_score > best_test_score) {
                best_test_score = test_score;
                best_test_epoch = epoch;
            }
        }

        std::cout << "Epoch " << epoch + 1 << "/" << settings.num_epochs << " | Cost " << cost << " | Best Test Score " << best_test_score * 100 << "%" << " at " << best_test_epoch << "\n";
    }

    eval_on_test(test_dataset, ffnn);
    
    return 0;
}
/*
99.04% at 159000
Settings settings{
        .ffnn_shape = {IMG_SIZE, 1024, 512, 255, 128, 64, 32, 16, NUM_CLASSES},
        .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax},
        .cost_type = CostType::categorical_cross_entropy,
        .reg_type = RegularizationType::none,
        .num_epochs = 1000000,
        .seed = 1,
        .batch_size = 1000,
        .lr = 0.0005,
        .noise = 1,
        .chance_for_noise = 0.5
    };

98.88%
Settings settings{
        .ffnn_shape = {IMG_SIZE, 1024, 512, 255, 128, 64, 32, 16, NUM_CLASSES},
        .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::sigmoid},
        .cost_type = CostType::binary_cross_entropy,
        .reg_type = RegularizationType::none,
        .num_epochs = 10000,
        .seed = 1,
        .batch_size = 1000,
        .lr = 0.0005,
        .noise = 1,
        .chance_for_noise = 0.5
    };

98.09% - went to 5k generations
Settings settings{
    .ffnn_shape = {IMG_SIZE, 512, 255, 128, 64, 32, 16, NUM_CLASSES},
    .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::sigmoid},
    .cost_type = ::binary_cross_entropy,
    .reg_type = RegularizationType::none,
    .num_epochs = 10000,
    .seed = 1,
    .batch_size = 1000,
    .lr = 0.0005
};
*/