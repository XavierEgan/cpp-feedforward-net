// g++ -std=c++23 -O3 -o test ./MNIST/MNIST.cpp && ./test
// del test.exe && g++ -std=c++23 -fopenmp -O3 -o test ./MNIST/MNIST.cpp && test.exe

//g++ -std=c++23 -march=native -fopenmp -O0 -ggdb -fno-omit-frame-pointer -o test ./MNIST/MNIST.cpp && lldb test.exe

// compile command on mac for multithreading:
/*
clang++ -std=c++23 -O3 -march=native -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" ./MNIST/MNIST.cpp -L"$(brew --prefix libomp)/lib" -lomp -o test && ./test
*/

#include "../AdamOptimiser.hpp"
#include "../RegularizationType.hpp"
#include "../LrSchedulers.hpp"

#include <fstream>
#include <cmath>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <thread>
#include <iomanip>

constexpr int IMG_SIZE = 784;
constexpr int NUM_CLASSES = 10;
constexpr int TRAIN_SIZE = 60000;
constexpr int TEST_SIZE = 10000;

constexpr int DEBUG_PRINT_INTERVAL = 100;
constexpr int EVAL_PRINT_INTERVAL = 1000;


struct Dataset {
    std::vector<Eigen::MatrixXf> images;
    std::vector<Eigen::MatrixXf> labels;
};

struct Settings {
    int batch_size = 256;
    RegularizationType regularization_type = RegularizationType::none;
    nn_utils::LRSchedulerExponential lr_scheduler = nn_utils::LRSchedulerExponential::from_num_generation(1e-2, 1e-3, 30000);
    std::vector<size_t> layer_sizes =               {IMG_SIZE, 512, 256, 128, 64, 32, NUM_CLASSES};
    std::vector<ActivationFunc> activation_funcs =  {relu, relu, relu, relu, relu, softmax};
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
    images.reserve(TRAIN_SIZE);
    labels.reserve(TRAIN_SIZE);

    std::getline(data_file, line); // remvoe the header line

    int i = 0;

    while (std::getline(data_file, line)) {
        if (i  % DEBUG_PRINT_INTERVAL == 0) {
            std::cout << "Reading line " << i << " From file " << data_loc << std::endl;
        }
        i++;

        Eigen::MatrixXf label = Eigen::MatrixXf::Zero(NUM_CLASSES, 1);
        Eigen::MatrixXf image(IMG_SIZE, 1);

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream line_stream(line);

        int lab;
        line_stream >> lab;
        label(lab, 0) = 1.0f;

        for (int i = 0; i < IMG_SIZE; i++) {
            float pixel = 0;
            line_stream >> pixel;
            pixel /= 255;
            image(i, 0) = pixel;
        }

        images.push_back(image);
        labels.push_back(label);
    }

    images.shrink_to_fit();
    labels.shrink_to_fit();

    return Dataset{images, labels};
}

#include <immintrin.h>

void eval_on_test(Dataset& test_data, FFNN& ffnn) {
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

int main() {
    // subnormal floats are like 2.5x slower or so from testing, so we just turn them off
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    std::srand(0);

    Eigen::setNbThreads(std::thread::hardware_concurrency() / 2);
    std::cout << "num htreads: " << Eigen::nbThreads() << std::endl;

    Settings settings;
    Dataset train_data = read_data("MNIST/MNIST/mnist_train.csv");
    Dataset test_data = read_data("MNIST/MNIST/mnist_test.csv");

    FFNN ffnn = FFNN::from_random_he_scaling(
        settings.layer_sizes,
        settings.activation_funcs,
        settings.regularization_type
    );
    AdamOptimiser optimiser(ffnn, CostType::categorical_cross_entropy);
    
    Eigen::MatrixXf minibatch;
    Eigen::MatrixXf minibatch_target;

    int gen = 0;
    while (!settings.lr_scheduler.is_done()) {
        auto start_time = std::chrono::high_resolution_clock::now();

        nn_utils::get_random_batch(train_data.images, train_data.labels, minibatch, minibatch_target, settings.batch_size);

        optimiser.lr = settings.lr_scheduler.lr;
        float avg_cost = optimiser.step(minibatch, minibatch_target);
        settings.lr_scheduler.step();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        if (gen % DEBUG_PRINT_INTERVAL == 0) {
            std::cout << "Generation " << std::fixed << std::setprecision(6) << gen << "  Avg Cost: " << avg_cost << "  Generation Time: " << elapsed.count() << " seconds " << "  Learning Rate: " << optimiser.lr << std::endl;
        }
        if (gen % EVAL_PRINT_INTERVAL == 0) {
            ffnn.write_to_file("mnist_ffnn_" + std::to_string(gen) + ".dat");
            eval_on_test(test_data, ffnn);
        }

        gen++;
    }

    eval_on_test(test_data, ffnn);

    ffnn.write_to_file("mnist_ffnn_final.dat");
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