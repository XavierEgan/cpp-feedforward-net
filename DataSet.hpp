#pragma once

#include "Eigen/Dense"

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>

// binary format:
//   [4 bytes] magic "FFDS"
//   [uint32]  version
//   [uint64]  n_samples
//   [int32]   input_rows, label_rows
//   [float * input_rows * n_samples] input data (column-major)
//   [float * label_rows * n_samples] label data (column-major)

// a full dataset stored as two big matrices, columns are samples
struct DataSet {
    Eigen::MatrixXf inputs;   // input_dim x n
    Eigen::MatrixXf labels;   // label_dim x n

    size_t size() const {
        return static_cast<size_t>(inputs.cols());
    }

    static DataSet empty(size_t input_dim, size_t label_dim) {
        return DataSet(Eigen::MatrixXf(input_dim, 0), Eigen::MatrixXf(label_dim, 0));
    }

    static DataSet from_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("from_file: cannot open file for reading: " + path);

        char magic[4];
        f.read(magic, 4);
        if (!f || magic[0] != 'F' || magic[1] != 'F' || magic[2] != 'D' || magic[3] != 'S') {
            throw std::runtime_error("from_file: not a valid DataSet file: " + path);
        }

        uint32_t version;
        f.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 2) {
            throw std::runtime_error("from_file: unsupported DataSet version in " + path);
        }

        uint64_t n;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));

        int32_t input_rows, label_rows;
        f.read(reinterpret_cast<char*>(&input_rows), sizeof(input_rows));
        f.read(reinterpret_cast<char*>(&label_rows), sizeof(label_rows));

        Eigen::MatrixXf inputs(input_rows, n);
        Eigen::MatrixXf labels(label_rows, n);

        f.read(reinterpret_cast<char*>(inputs.data()), inputs.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(labels.data()), labels.size() * sizeof(float));

        if (!f) throw std::runtime_error("from_file: read failed: " + path);

        return DataSet(std::move(inputs), std::move(labels));
    }

    static DataSet from_files(const std::vector<std::string>& paths) {
        if (paths.empty()) {
            throw std::invalid_argument("from_files: paths cannot be empty");
        }

        DataSet data = from_file(paths.front());
        for (size_t i = 1; i < paths.size(); i++) {
            data.append(from_file(paths.at(i)));
        }
        return data;
    }

    static DataSet from_matrices(Eigen::MatrixXf inputs, Eigen::MatrixXf labels) {
        return DataSet(std::move(inputs), std::move(labels));
    }

    // builds a dataset from a vector of per-sample column vectors, e.g. from a self-play loop
    // that doesn't know the sample count ahead of time
    static DataSet from_samples(const std::vector<Eigen::VectorXf>& inputs, const std::vector<Eigen::VectorXf>& labels) {
        if (inputs.size() != labels.size()) {
            throw std::invalid_argument("from_samples: inputs and labels must be the same size");
        }
        if (inputs.empty()) {
            throw std::invalid_argument("from_samples: inputs cannot be empty");
        }

        Eigen::MatrixXf input_mat(inputs.front().size(), inputs.size());
        Eigen::MatrixXf label_mat(labels.front().size(), labels.size());

        for (size_t i = 0; i < inputs.size(); i++) {
            input_mat.col(i) = inputs.at(i);
            label_mat.col(i) = labels.at(i);
        }

        return DataSet(std::move(input_mat), std::move(label_mat));
    }

    void write_to_file(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("write_to_file: cannot open file for writing: " + path);

        f.write("FFDS", 4);

        const uint32_t version = 2;
        f.write(reinterpret_cast<const char*>(&version), sizeof(version));

        const uint64_t n = static_cast<uint64_t>(inputs.cols());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));

        const int32_t input_rows = static_cast<int32_t>(inputs.rows());
        const int32_t label_rows = static_cast<int32_t>(labels.rows());
        f.write(reinterpret_cast<const char*>(&input_rows), sizeof(input_rows));
        f.write(reinterpret_cast<const char*>(&label_rows), sizeof(label_rows));

        f.write(reinterpret_cast<const char*>(inputs.data()), inputs.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(labels.data()), labels.size() * sizeof(float));

        if (!f) throw std::runtime_error("write_to_file: write failed: " + path);
    }

    void append(const DataSet& other) {
        if (other.inputs.rows() != inputs.rows() || other.labels.rows() != labels.rows()) {
            throw std::invalid_argument("append: shape mismatch");
        }

        const Eigen::Index old_n = inputs.cols();
        const Eigen::Index add_n = other.inputs.cols();

        inputs.conservativeResize(Eigen::NoChange, old_n + add_n);
        inputs.rightCols(add_n) = other.inputs;

        labels.conservativeResize(Eigen::NoChange, old_n + add_n);
        labels.rightCols(add_n) = other.labels;
    }

    DataSet() = delete;
private:
    DataSet(Eigen::MatrixXf inputs, Eigen::MatrixXf labels) : inputs(std::move(inputs)), labels(std::move(labels)) {
        if (this->inputs.cols() != this->labels.cols()) {
            throw std::invalid_argument("DataSet: inputs and labels must have the same number of samples");
        }
    }
};

// draws random mini-batches from a dataset; the rng is seeded once at construction, so the
// same seed always reproduces the same sequence of batches
struct Batcher {
    const DataSet& data;
    std::mt19937 rng;
    std::vector<Eigen::Index> indices;

    static Batcher from_dataset(const DataSet& data, unsigned int seed = std::random_device{}()) {
        return Batcher(data, seed);
    }

    void next_batch(size_t batch_size, Eigen::MatrixXf& out_inputs, Eigen::MatrixXf& out_labels) {
        const size_t n = data.size();
        if (n == 0) {
            throw std::invalid_argument("next_batch: dataset cannot be empty");
        }
        if (batch_size > n) batch_size = n;

        if (indices.size() != n) {
            indices.resize(n);
            std::iota(indices.begin(), indices.end(), 0);
        }

        std::shuffle(indices.begin(), indices.end(), rng);

        out_inputs.resize(data.inputs.rows(), batch_size);
        out_labels.resize(data.labels.rows(), batch_size);

        for (size_t b = 0; b < batch_size; b++) {
            out_inputs.col(b) = data.inputs.col(indices.at(b));
            out_labels.col(b) = data.labels.col(indices.at(b));
        }
    }

    Batcher() = delete;
private:
    Batcher(const DataSet& data, unsigned int seed) : data(data), rng(seed) {}
};
