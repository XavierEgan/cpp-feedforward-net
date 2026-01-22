// g++ -std=c++23 -march=native -O3 -o test ./MNIST/MNIST.cpp && ./test  

#include "../FFNN.cpp"
#include <fstream>

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

    while (std::getline(data_file, line)) {
        Eigen::MatrixXf label = Eigen::MatrixXf::Zero(10, 1);
        Eigen::MatrixXf image(784, 1);

        std::istringstream line_stream(line);
        std::string dummy;

        int lab;
        line_stream >> lab;
        label(lab, 0) = 1.0f;

        line_stream >> dummy;

        for (int i = 0; i < 784; i++) {
            line_stream >> image(i, 0);
            line_stream >> dummy;
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
        {784, 256, 128, 64, 10},
        {ActivationFunc::relu,  ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::linear}
    );

    const int num_generations = 1000;
    double lr = 0.005;
    const double decay = 0.999;
    const int batch_size = 10;

    for (int gen = 0; gen < 1000; gen++) {
        auto avg_cost = ffnn.gradient_descent(train_data.images, train_data.labels, batch_size, lr);
        lr *= decay;

        std::cout << "Generation " << gen << "  Avg Cost: " << avg_cost << "  lr: " << lr << std::endl;
    }

    // check to see how good our model is
    int total_right = 0;
    for (int i = 0; i < test_data.images.size(); i++) {
        
    }
}