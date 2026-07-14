#include "../DataSet.hpp"
#include "../AdamOptimiser.hpp"
#include "../Trainer.hpp"
#include "../LrSchedulers.hpp"

#include <string>
#include <filesystem>
#include <random>
#include <iostream>
#include <vector>

constexpr int NOISE_SIZE = 10;
constexpr int IMG_SIZE = 784;

void print_image(const Eigen::MatrixXf& image) {
    for (Eigen::Index i = 0; i < 28; i++) {
        for (Eigen::Index j = 0; j < 28; j++) {
            std::cout << (image(i * 28 + j, 0) > 0.5f ? "██" : "░░");
        }
        std::cout << "\n";
    }
}

constexpr float INPUT_NOISE_STDDEV = 0.1f;

int main() {
    // one-hot vector for each digit, used both as the generator's input and the adversary's
    // target - digit_to_input.at(d) is what "asking for a d" looks like
    std::vector<Eigen::MatrixXf> digit_to_input(NOISE_SIZE);
    for (int digit = 0; digit < NOISE_SIZE; digit++) {
        Eigen::MatrixXf input = Eigen::MatrixXf::Zero(NOISE_SIZE, 1);
        input.row(digit).setOnes();
        digit_to_input.at(digit) = input;
    }

    FFNN adversary = FFNN::from_file("MNIST/models/MNIST-Cirrus-1.dat");
    BackpropWorkspace adversary_ws = BackpropWorkspace::from_shape(adversary.network_shape);

    FFNN image_gen = FFNN::from_random_he_scaling(
        {NOISE_SIZE, 512, 1024, IMG_SIZE},
        {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::linear}
    );

    AdamOptimiser optimiser = AdamOptimiser::from_ffnn(image_gen, CostType::categorical_cross_entropy, 0.01, RegularizationType::none);

    constexpr size_t num_steps = 1000;
    constexpr size_t batch_size = 128;

    std::mt19937 rng(1);
    std::normal_distribution<float> noise_dist(0.0f, INPUT_NOISE_STDDEV);
    std::uniform_int_distribution<int> digit_dist(0, NOISE_SIZE - 1);

    Eigen::MatrixXf target(NOISE_SIZE, batch_size);
    Eigen::MatrixXf input(NOISE_SIZE, batch_size);

    for (size_t step = 0; step < num_steps; step++) {
        // each column gets a randomly chosen digit's one-hot vector, plus a little noise so the
        // generator can't just memorize one fixed image per digit
        for (size_t b = 0; b < batch_size; b++) {
            const int digit = digit_dist(rng);
            target.col(b) = digit_to_input.at(digit);
            input.col(b) = digit_to_input.at(digit);
        }
        for (Eigen::Index i = 0; i < input.size(); i++) input.data()[i] += noise_dist(rng);

        const Eigen::MatrixXf fake_images = image_gen.forward(input, optimiser.ws.fwd);
        const Eigen::MatrixXf& adversary_output = adversary.forward(fake_images, adversary_ws.fwd);

        const Eigen::MatrixXf adversary_output_delta = adversary_output - target;
        const float cost = nn_utils::cost(adversary_output, target, CostType::categorical_cross_entropy);

        const Eigen::MatrixXf grad_wrt_fake_images = adversary.backward_to_input(fake_images, adversary_output_delta, adversary_ws);

        optimiser.step_from_output_delta(input, grad_wrt_fake_images);

        if (step % 100 == 0) {
            std::cout << "step " << step << "/" << num_steps << " | cost " << cost << "\n";
        }
    }

    for (int digit = 0; digit < NOISE_SIZE; digit++) {
        std::cout << "digit " << digit << ":\n";
        print_image(image_gen.forward(digit_to_input.at(digit), optimiser.ws.fwd));
    }

    image_gen.write_to_file("image-gen/generator.dat");
}