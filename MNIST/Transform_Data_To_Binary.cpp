#include "../AdamOptimiser.hpp"
#include "../DataSet.hpp"

#include <fstream>

DataSet read_data(const std::string& data_loc) {
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
    std::cout << images.size() << ", " << labels.size() << "\n";
    return DataSet{images, labels};
}


int main() {
    DataSet test_data = read_data("MNIST/MNIST/test.csv");
    DataSet train_data = read_data("MNIST/MNIST/train.csv");

    DataSet fashion_test_data = read_data("MNIST/Fashion-MNIST/test.csv");
    DataSet fashion_train_data = read_data("MNIST/Fashion-MNIST/train.csv");

    test_data.write("MNIST/MNIST/bin/test.dat");
    train_data.write("MNIST/MNIST/bin/train.dat");

    fashion_test_data.write("MNIST/Fashion-MNIST/bin/test.dat");
    fashion_train_data.write("MNIST/Fashion-MNIST/bin/train.dat");
}