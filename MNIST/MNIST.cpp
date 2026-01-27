// g++ -std=c++23 -march=native -O3 -o test ./MNIST/MNIST.cpp && ./test
// g++ -std=c++23 -march=native -fopenmp -O3 -o test ./MNIST/MNIST.cpp && test.exe

// compile command on mac for multithreading:
/*
clang++ -std=c++23 -O3 -march=native -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" ./MNIST/MNIST.cpp -L"$(brew --prefix libomp)/lib" -lomp -o test && ./test
*/

#include "../FFNN.cpp"
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
    std::getline(data_file, line); // remvoe the header line

    while (std::getline(data_file, line)) {
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

int main() {
    Eigen::setNbThreads(std::thread::hardware_concurrency() / 2);

    std::cout << "num htreads: " << Eigen::nbThreads() << std::endl;

    const int max_generations = 276447231;
    const float target_cost = 0.0;
    const int batch_size = 250;
    const float noise_amplitude = 0.1f;

    DecayOnPlateauScheduler scheduler(0.2, 0.001, 0.996, 20);

    Dataset train_data = read_data("MNIST/MNIST/mnist_train.csv");
    Dataset test_data = read_data("MNIST/MNIST/mnist_test.csv");

    FFNN ffnn = FFNN::from_random(
        {784, 1028, 2048, 1028, 512, 256, 128, 64, 32, 10},
        {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::sigmoid},
        CostType::binary_cross_entropy
    );

    int gen = 0;
    float avg_cost = 1000.0f;
    auto start_time = std::chrono::high_resolution_clock::now();
    while (avg_cost > target_cost && gen < max_generations) {
        start_time = std::chrono::high_resolution_clock::now();

        auto [minibatch, minibatch_target] = FFNN::get_batch(train_data.images, train_data.labels, batch_size);

        minibatch += Eigen::MatrixXf::Random(minibatch.rows(), minibatch.cols()) * noise_amplitude;
        minibatch = minibatch.cwiseMin(1.0f).cwiseMax(0.0f);

        avg_cost = ffnn.gradient_descent(minibatch, minibatch_target, scheduler.learning_rate);

        bool done_training = scheduler.step(avg_cost);
        if (done_training) {
            break;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        if (gen % 10 == 0) {
            std::cout << "Generation " << std::fixed << std::setprecision(6) << gen + 1 << "  Avg Cost: " << avg_cost << "  lr: " << scheduler.learning_rate << "  Generation Time: " << elapsed.count() << " seconds\n";
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