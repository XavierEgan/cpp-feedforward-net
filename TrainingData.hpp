#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include "Eigen/Dense"

// Binary format:
//   [uint64] n_samples
//   [int32]  input_rows, input_cols
//   [int32]  label_rows, label_cols
//   [float * input_rows * input_cols] input data (column-major) -- repeated n_samples times
//   [float * label_rows * label_cols] label data (column-major) -- repeated n_samples times

class TrainingData {
public:
    std::vector<Eigen::MatrixXf> inputs;
    std::vector<Eigen::MatrixXf> labels;

    void write(const std::string& path) const {
        if (inputs.size() != labels.size())
            throw std::runtime_error("TrainingData: inputs and labels size mismatch");

        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("TrainingData: cannot open file for writing: " + path);

        uint64_t n = static_cast<uint64_t>(inputs.size());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));

        int32_t ir = 0, ic = 0, lr = 0, lc = 0;
        if (n > 0) {
            ir = static_cast<int32_t>(inputs[0].rows());
            ic = static_cast<int32_t>(inputs[0].cols());
            lr = static_cast<int32_t>(labels[0].rows());
            lc = static_cast<int32_t>(labels[0].cols());
        }
        f.write(reinterpret_cast<const char*>(&ir), sizeof(ir));
        f.write(reinterpret_cast<const char*>(&ic), sizeof(ic));
        f.write(reinterpret_cast<const char*>(&lr), sizeof(lr));
        f.write(reinterpret_cast<const char*>(&lc), sizeof(lc));

        for (uint64_t i = 0; i < n; ++i) {
            f.write(reinterpret_cast<const char*>(inputs[i].data()),
                    inputs[i].size() * sizeof(float));
        }
        for (uint64_t i = 0; i < n; ++i) {
            f.write(reinterpret_cast<const char*>(labels[i].data()),
                    labels[i].size() * sizeof(float));
        }

        if (!f) throw std::runtime_error("TrainingData: write failed: " + path);
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("TrainingData: cannot open file for reading: " + path);

        uint64_t n;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));

        int32_t ir, ic, lr, lc;
        f.read(reinterpret_cast<char*>(&ir), sizeof(ir));
        f.read(reinterpret_cast<char*>(&ic), sizeof(ic));
        f.read(reinterpret_cast<char*>(&lr), sizeof(lr));
        f.read(reinterpret_cast<char*>(&lc), sizeof(lc));

        inputs.resize(n);
        labels.resize(n);

        for (uint64_t i = 0; i < n; ++i) {
            inputs[i].resize(ir, ic);
            f.read(reinterpret_cast<char*>(inputs[i].data()),
                   ir * ic * sizeof(float));
        }
        for (uint64_t i = 0; i < n; ++i) {
            labels[i].resize(lr, lc);
            f.read(reinterpret_cast<char*>(labels[i].data()),
                   lr * lc * sizeof(float));
        }

        if (!f && !f.eof())
            throw std::runtime_error("TrainingData: read failed: " + path);
    }
};
