#include "../AdamOptimiser.hpp"

#include <fstream>

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


int main() {
    FFNN ffnn = FFNN::from_file("mnist_ffnn.dat");

    Dataset test_data = read_data("MNIST/MNIST/mnist_test.csv");

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