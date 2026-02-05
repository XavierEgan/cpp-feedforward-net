// g++ -std=c++23 -O3 -o test ./MNIST/MNIST.cpp && ./test
// del test.exe && g++ -std=c++23 -fopenmp -O3 -o test ./MNIST/MNIST.cpp && test.exe

//g++ -std=c++23 -march=native -fopenmp -O0 -ggdb -fno-omit-frame-pointer -o test ./MNIST/MNIST.cpp && lldb test.exe

// compile command on mac for multithreading:
/*
clang++ -std=c++23 -O3 -march=native -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" ./MNIST/MNIST.cpp -L"$(brew --prefix libomp)/lib" -lomp -o test && ./test
*/

#include "../AdamOptimiser.hpp"

#include <fstream>
#include <cmath>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <thread>
#include <iomanip>

struct Dataset {
    std::vector<Eigen::MatrixXf> images;
    std::vector<Eigen::MatrixXf> labels;
};

Dataset read_data(const std::string& data_loc) {
    std::ifstream data_file(data_loc);

    if (!data_file) {
        std::cout << "failed to read file" << std::endl;
        throw std::runtime_error("read_data: failed to read file: " + data_loc);
    }

    std::string line;

    std::vector<Eigen::MatrixXf> images;
    std::vector<Eigen::MatrixXf> labels;
    images.reserve(60000);
    labels.reserve(60000);

    std::getline(data_file, line); // remvoe the header line

    int i = 0;

    while (std::getline(data_file, line)) {
        if (i  % 100 == 0) {
            std::cout << "Reading line " << i << " From file " << data_loc << std::endl;
        }
        if (i  == 1000) {
            //return Dataset{images, labels};
        }
        i++;

        Eigen::MatrixXf label = Eigen::MatrixXf::Zero(10, 1);
        Eigen::MatrixXf image(784, 1);

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream line_stream(line);

        int lab;
        line_stream >> lab;
        label(lab, 0) = 1.0f;

        for (int i = 0; i < 784; i++) {
            float pixel = 0;
            line_stream >> pixel;
            pixel /= 255;
            image(i, 0) = pixel;
        }

        images.push_back(image);
        labels.push_back(label);
    }
    return Dataset{images, labels};
}

#include <immintrin.h>

int main() {
    // subnormal floats are like 2.5x slower or so from testing, so we just turn them off
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    Eigen::setNbThreads(std::thread::hardware_concurrency() / 2);
    std::cout << "num htreads: " << Eigen::nbThreads() << std::endl;

    const int max_generations = 1000;
    const float target_cost = 0.0;
    const int batch_size = 500;
    const float noise_amplitude = 0.1f;

    Dataset train_data = read_data("MNIST/MNIST/mnist_train.csv");
    Dataset test_data = read_data("MNIST/MNIST/mnist_test.csv");

    std::vector<size_t> layer_sizes = {784, 512, 256, 128, 64, 32, 10};
    std::vector<ActivationFunc> activation_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax};

    FFNN ffnn = FFNN::from_random_he_scaling(
        layer_sizes,
        activation_funcs
    );

    AdamOptimiser optimiser(ffnn, CostType::categorical_cross_entropy, 0.001);

    int gen = 0;
    float avg_cost = 1000.0f;
    auto start_time = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXf minibatch;
    Eigen::MatrixXf minibatch_target;
    while (gen < max_generations && avg_cost > target_cost) {
        start_time = std::chrono::high_resolution_clock::now();
        
        nn_utils::get_batch(train_data.images, train_data.labels, minibatch, minibatch_target, batch_size);

        minibatch += Eigen::MatrixXf::Random(minibatch.rows(), minibatch.cols()) * noise_amplitude;
        minibatch = minibatch.cwiseMin(1.0f).cwiseMax(0.0f);

        avg_cost = optimiser.step(minibatch, minibatch_target);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        if (gen % 10 == 0) {
            std::cout << "Generation " << std::fixed << std::setprecision(6) << gen << "  Avg Cost: " << avg_cost << "  Generation Time: " << elapsed.count() << " seconds\n";
        }
        gen++;
    }

    int total_right = 0;
    for (int d = 0; d < (int)test_data.images.size(); d++) {
        // ground-truth = argmax(label)
        Eigen::Index gt_r = 0, gt_c = 0;
        test_data.labels[d].maxCoeff(&gt_r, &gt_c);
        const Eigen::Index gt = gt_r;

        // prediction = argmax(output)
        auto res = ffnn.forward(test_data.images[d]);
        Eigen::Index pred_r = 0, pred_c = 0;
        res.maxCoeff(&pred_r, &pred_c);
        const Eigen::Index pred = pred_r;

        if (pred == gt) total_right++;
    }

    std::cout << "Total test correct: " << total_right << " Percentage right: " << (static_cast<float>(total_right) / static_cast<float>(test_data.images.size())) * 100.0f << "%" << std::endl;
}

/*
98.61%
FFNN ffnn = FFNN::from_random(
    {784, 512, 256, 128, 64, 32, 10},
    {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::sigmoid},
    CostType::binary_cross_entropy
);
DecayOnPlateauScheduler scheduler(0.5, 0.001, 0.99, 30);
batch size = 250
*/
/*

*/