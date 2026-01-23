// g++ -std=c++23 -march=native -O3 -o test ./MNIST/MNIST.cpp && ./test  

#include "../FFNN.cpp"
#include <fstream>
#include <cmath>
#include <chrono>

struct Dataset {
    std::vector<Eigen::MatrixXf> images;
    std::vector<Eigen::MatrixXf> labels;
};

Dataset read_data(const std::string& data_loc) {
    std::ifstream data_file(data_loc);

    if (!data_file) {
        std::cout << "failed to read file" << std::endl;
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
    Dataset train_data = read_data("MNIST/archive/mnist_train.csv");
    Dataset test_data = read_data("MNIST/archive/mnist_test.csv");

    FFNN ffnn = FFNN::from_random(
        {784, 512, 256, 128, 64, 32, 10},
        {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::sigmoid},
        CostType::binary_cross_entropy
    );

    const int max_generations = 1000;
    const float target_cost = 0.0;

    double lr = 0.5;
    const double min_lr = 0.005;
    const double decay = 0.999;
    const int batch_size = 1000;

    int gen = 0;
    float avg_cost = 1000.0f;
    auto start_time = std::chrono::high_resolution_clock::now();
    while (avg_cost > target_cost and gen < max_generations) {
        start_time = std::chrono::high_resolution_clock::now();

        avg_cost = ffnn.gradient_descent(train_data.images, train_data.labels, batch_size, lr);
        lr *= decay;
        lr = std::max(lr, min_lr);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Generation " << gen + 1 << "  Avg Cost: " << avg_cost << "  lr: " << lr << "  Generation Time: " << elapsed.count() << " seconds" << std::endl;
        gen++;
    }

    // check to see how good our model is
    int total_right = 0;
    for (int d = 0; d < test_data.images.size(); d++) {
        int cor_ans = -1;
        for (int i = 0; i < 10; i++) {
            if (test_data.labels.at(d)(i, 0) > .9) {
                cor_ans = i;
                break;
            }
        }

        auto res = ffnn.forward(test_data.images.at(d));
        for (int i = 0; i < 10; i++) {
            if (round(res.maxCoeff()) == 1 && i == cor_ans) {
                total_right += 1;
                break;
            };
        }
    }

    std::cout << "Total test correct: " << total_right << " Percentage right: " << (static_cast<float>(total_right) / static_cast<float>(test_data.images.size())) * 100.0f << "%" << std::endl;
}