// clang++ -std=c++23 -O0 -I.. tests/test_serialization.cpp -o test_serialization && ./test_serialization

#include "../FFNN.hpp"
#include "../DataSet.hpp"

#include <iostream>
#include <cassert>
#include <filesystem>

void test_ffnn_round_trip() {
    FFNN ffnn = FFNN::from_random_he_scaling({4, 6, 3}, {ActivationFunc::relu, ActivationFunc::softmax});
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(4, 5);
    Eigen::MatrixXf out_before = ffnn.forward(input);

    const std::string path = "/tmp/ffnn_roundtrip_test.dat";
    ffnn.write_to_file(path);
    FFNN loaded = FFNN::from_file(path);
    Eigen::MatrixXf out_after = loaded.forward(input);

    assert((out_before - out_after).norm() < 1e-6f);
    assert(loaded.network_shape == ffnn.network_shape);
    assert(loaded.activation_functions == ffnn.activation_functions);
    std::filesystem::remove(path);
    std::cout << "FFNN round trip: PASS\n";
}

void test_ffnn_rejects_bad_magic() {
    const std::string path = "/tmp/ffnn_bad_magic_test.dat";
    std::ofstream bad(path, std::ios::binary);
    bad.write("XXXX", 4);
    bad.close();

    bool threw = false;
    try {
        FFNN::from_file(path);
    } catch (const std::exception&) {
        threw = true;
    }
    std::filesystem::remove(path);

    assert(threw);
    std::cout << "FFNN rejects bad magic: PASS\n";
}

void test_dataset_round_trip() {
    std::vector<Eigen::VectorXf> inputs, labels;
    for (int i = 0; i < 10; i++) {
        Eigen::VectorXf in(3); in << i, i * 2, i * 3;
        Eigen::VectorXf out(1); out << i * 10;
        inputs.push_back(in);
        labels.push_back(out);
    }
    DataSet ds = DataSet::from_samples(inputs, labels);

    const std::string path = "/tmp/dataset_roundtrip_test.dat";
    ds.write_to_file(path);
    DataSet loaded = DataSet::from_file(path);
    std::filesystem::remove(path);

    assert(loaded.size() == ds.size());
    assert((loaded.inputs - ds.inputs).norm() < 1e-6f);
    assert((loaded.labels - ds.labels).norm() < 1e-6f);
    std::cout << "DataSet round trip: PASS\n";
}

int main() {
    test_ffnn_round_trip();
    test_ffnn_rejects_bad_magic();
    test_dataset_round_trip();

    std::cout << "ALL PASS\n";
    return 0;
}
