// clang++ -std=c++23 -O0 -I.. tests/test_batcher.cpp -o test_batcher && ./test_batcher

#include "../DataSet.hpp"

#include <iostream>
#include <cassert>

DataSet make_dataset(int n) {
    std::vector<Eigen::VectorXf> inputs, labels;
    for (int i = 0; i < n; i++) {
        Eigen::VectorXf in(2); in << i, i * 2;
        Eigen::VectorXf out(1); out << i;
        inputs.push_back(in);
        labels.push_back(out);
    }
    return DataSet::from_samples(inputs, labels);
}

void test_same_seed_same_sequence() {
    DataSet ds = make_dataset(20);
    Batcher b1 = Batcher::from_dataset(ds, 42);
    Batcher b2 = Batcher::from_dataset(ds, 42);

    Eigen::MatrixXf i1, l1, i2, l2;
    for (int call = 0; call < 3; call++) {
        b1.next_batch(6, i1, l1);
        b2.next_batch(6, i2, l2);
        assert((i1 - i2).norm() < 1e-6f);
        assert((l1 - l2).norm() < 1e-6f);
    }
    std::cout << "same seed -> same sequence: PASS\n";
}

void test_different_seed_different_sequence() {
    DataSet ds = make_dataset(20);
    Batcher b1 = Batcher::from_dataset(ds, 1);
    Batcher b2 = Batcher::from_dataset(ds, 2);

    Eigen::MatrixXf i1, l1, i2, l2;
    b1.next_batch(6, i1, l1);
    b2.next_batch(6, i2, l2);
    assert((i1 - i2).norm() > 1e-3f);
    std::cout << "different seed -> different sequence: PASS\n";
}

void test_batch_size_clamps_to_dataset_size() {
    DataSet ds = make_dataset(5);
    Batcher b = Batcher::from_dataset(ds, 7);

    Eigen::MatrixXf inputs, labels;
    b.next_batch(1000, inputs, labels);
    assert(inputs.cols() == 5);
    assert(labels.cols() == 5);
    std::cout << "batch_size clamps to dataset size: PASS\n";
}

int main() {
    test_same_seed_same_sequence();
    test_different_seed_different_sequence();
    test_batch_size_clamps_to_dataset_size();

    std::cout << "ALL PASS\n";
    return 0;
}
