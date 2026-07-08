/*
clang++ -std=c++23 -O3 -march=native ./spiral/spiral.cpp -o spiral_demo && ./spiral_demo
*/

#include "../DataSet.hpp"
#include "../AdamOptimiser.hpp"

#include <cmath>
#include <random>
#include <iostream>
#include <string>
#include <vector>

constexpr int NUM_CLASSES = 3;
constexpr int POINTS_PER_CLASS = 400;
constexpr float TEST_SPLIT = 0.2f;

constexpr int EVAL_PRINT_INTERVAL = 200;

// ansi colours for the decision boundary render, one per class
const std::string CLASS_COLOURS[NUM_CLASSES] = {"\033[31m", "\033[32m", "\033[34m"};
const std::string COLOUR_RESET = "\033[0m";

struct Settings {
    std::vector<size_t> ffnn_shape;
    std::vector<ActivationFunc> ffnn_funcs;
    CostType cost_type;
    RegularizationType reg_type;

    size_t num_epochs;
    int seed;
    size_t batch_size;
    double lr;
    float spiral_noise;
};

/*
generates the classic n-arm spiral dataset

each class is one arm of a spiral: points start near the origin and
wind outwards, with gaussian noise added to the angle so the arms
overlap a little near the centre

inputs are 2x1 (x, y) in roughly [-1, 1], labels are one-hot NUM_CLASSES x 1
*/
void generate_spiral(std::vector<Eigen::VectorXf>& train_inputs, std::vector<Eigen::VectorXf>& train_labels,
                      std::vector<Eigen::VectorXf>& test_inputs, std::vector<Eigen::VectorXf>& test_labels,
                      float noise, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> angle_noise(0.0f, noise);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    for (int c = 0; c < NUM_CLASSES; c++) {
        for (int i = 0; i < POINTS_PER_CLASS; i++) {
            // radius grows from 0 to 1 along the arm
            const float r = static_cast<float>(i) / POINTS_PER_CLASS;

            // each arm is offset by an equal share of the full turn
            const float t = r * 4.0f + (static_cast<float>(c) / NUM_CLASSES) * 2.0f * static_cast<float>(M_PI) + angle_noise(rng);

            Eigen::VectorXf input(2);
            input(0) = r * std::sin(t);
            input(1) = r * std::cos(t);

            Eigen::VectorXf label = Eigen::VectorXf::Zero(NUM_CLASSES);
            label(c) = 1.0f;

            if (unit(rng) < TEST_SPLIT) {
                test_inputs.push_back(input);
                test_labels.push_back(label);
            } else {
                train_inputs.push_back(input);
                train_labels.push_back(label);
            }
        }
    }
}

float eval_accuracy(const DataSet& data, const FFNN& ffnn) {
    int total_right = 0;

    // get all predictions in one forward pass
    const auto predictions = ffnn.forward(data.inputs);

    for (Eigen::Index i = 0; i < data.inputs.cols(); i++) {
        Eigen::Index answer = 0, answer_col = 0;
        data.labels.col(i).maxCoeff(&answer, &answer_col);

        Eigen::Index prediction = 0, prediction_col = 0;
        predictions.col(i).maxCoeff(&prediction, &prediction_col);

        if (answer == prediction) total_right++;
    }

    return static_cast<float>(total_right) / static_cast<float>(data.size());
}

/*
renders the learned decision boundary as a coloured ascii grid

builds a grid of points over [-1.1, 1.1]^2, runs them all through the
network in a single forward pass, and prints one coloured character per
cell based on the predicted class
*/
void render_decision_boundary(const FFNN& ffnn, int width = 60, int height = 30) {
    const float lo = -1.1f, hi = 1.1f;

    // one column per grid cell
    Eigen::MatrixXf grid(2, width * height);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            const float x = lo + (hi - lo) * (static_cast<float>(col) / (width - 1));
            const float y = hi - (hi - lo) * (static_cast<float>(row) / (height - 1));
            grid(0, row * width + col) = x;
            grid(1, row * width + col) = y;
        }
    }

    const auto predictions = ffnn.forward(grid);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            Eigen::Index prediction = 0, prediction_col = 0;
            predictions.col(row * width + col).maxCoeff(&prediction, &prediction_col);

            // shade by confidence: strong prediction gets a solid block
            const float confidence = predictions(prediction, row * width + col);
            const char* shade = confidence > 0.9f ? "#" : (confidence > 0.6f ? "+" : ".");

            std::cout << CLASS_COLOURS[prediction] << shade;
        }
        std::cout << COLOUR_RESET << "\n";
    }
}

int main() {
    Settings settings{
        .ffnn_shape = {2, 64, 64, 32, NUM_CLASSES},
        .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax},
        .cost_type = CostType::categorical_cross_entropy,
        .reg_type = RegularizationType::none,
        .num_epochs = 4000,
        .seed = 1,
        .batch_size = 128,
        .lr = 0.002,
        .spiral_noise = 0.2f
    };

    srand(settings.seed);

    // generate the dataset in-memory, no files needed
    std::vector<Eigen::VectorXf> train_inputs, train_labels, test_inputs, test_labels;
    generate_spiral(train_inputs, train_labels, test_inputs, test_labels, settings.spiral_noise, settings.seed);

    DataSet train_dataset = DataSet::from_samples(train_inputs, train_labels);
    DataSet test_dataset = DataSet::from_samples(test_inputs, test_labels);

    std::cout << "train samples: " << train_dataset.size() << " | test samples: " << test_dataset.size() << "\n";

    FFNN ffnn = FFNN::from_random_he_scaling(settings.ffnn_shape, settings.ffnn_funcs);
    AdamOptimiser optimiser = AdamOptimiser::from_ffnn(ffnn, settings.cost_type, settings.lr, settings.reg_type);
    Batcher batcher = Batcher::from_dataset(train_dataset, settings.seed);

    Eigen::MatrixXf inputs;
    Eigen::MatrixXf targets;

    float best_test_score = std::numeric_limits<float>::min();
    int best_test_epoch = 0;

    for (auto epoch{0uz}; epoch < settings.num_epochs; epoch++) {
        batcher.next_batch(settings.batch_size, inputs, targets);

        float cost = optimiser.step(inputs, targets);

        if (epoch % EVAL_PRINT_INTERVAL == 0) {
            const float test_score = eval_accuracy(test_dataset, ffnn);
            if (test_score > best_test_score) {
                best_test_score = test_score;
                best_test_epoch = epoch;
            }
            std::cout << "Epoch " << epoch + 1 << "/" << settings.num_epochs << " | Cost " << cost << " | Test " << test_score * 100 << "% | Best " << best_test_score * 100 << "% at " << best_test_epoch << "\n";
        }
    }

    const float final_train = eval_accuracy(train_dataset, ffnn);
    const float final_test = eval_accuracy(test_dataset, ffnn);
    std::cout << "\nfinal train accuracy: " << final_train * 100 << "%\n";
    std::cout << "final test accuracy:  " << final_test * 100 << "%\n\n";

    render_decision_boundary(ffnn);

    ffnn.write_to_file("spiral/spiral_model.dat");
    std::cout << "\nmodel saved to spiral/spiral_model.dat\n";

    return 0;
}
