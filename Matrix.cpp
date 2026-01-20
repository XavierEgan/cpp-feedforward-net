// g++ -std=c++23 -o test Matrix.cpp && test.exe
// g++ -std=c++23 -o test Matrix.cpp && gdb test.exe

// g++ -std=c++23 -g3 -ggdb -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o test.exe Matrix.cpp && test.exe
// g++ -std=c++23 -g3 -ggdb -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o test.exe Matrix.cpp && gdb test.exe

#include <memory>
#include <iostream>
#include <string.h>
#include <chrono>
#include <vector>
#include <string>

struct Matrix {
    size_t width;
    size_t height;
    std::vector<float> data;

    Matrix() {}

    Matrix(size_t height, size_t width) noexcept : width(width), height(height) {
        data.resize(width * height);
    }

    Matrix(size_t height, size_t width, const std::vector<float>& values) noexcept : width(width), height(height) {
        data = values;
    }

    static Matrix zeros(size_t height, size_t width) {
        Matrix m(height, width);
        for (int i = 0; i < height * width; i++) {
            m.data[i] = 0.0;
        }
        return m;
    }

    static Matrix from_random(size_t height, size_t width, float lo = -1.0, float hi = 1.0) {
        Matrix m(height, width);
        for (int i = 0; i < height * width; i++) {
            m.data[i] = (((float)rand()) / ((float)RAND_MAX)) * (hi - lo) + lo;
        }
        return m;
    }

    float& at(size_t y, size_t x) noexcept {
        return data[y * width + x];
    }

    const float& at(size_t y, size_t x) const noexcept {
        return data[y * width + x];
    }

    friend Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
        if (lhs.width != rhs.height) {
            std::cout << "size mismatch in mult " << lhs.get_dim() << "and" << rhs.get_dim() << std::endl;
        }

        Matrix m = Matrix::zeros(lhs.height, rhs.width);

        for (size_t y = 0; y < lhs.height; y++) {
            for (size_t x = 0; x < rhs.width; x++) {
                float s = 0;
                for (size_t i = 0; i < lhs.width; i++) {
                    s += lhs.at(y, i) * rhs.at(i, x);
                }
                m.at(y, x) = s;
            }
        }

        return m;
    }

    friend Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
        if (lhs.width != rhs.width || lhs.height != rhs.height) {
            std::cout << "sizes dont match for +. sizes " << "(" << lhs.height << ", " << lhs.width << ")" << " (" << rhs.height << ", " << rhs.width << ")" << std::endl;
        }

        Matrix m = Matrix::zeros(lhs.height, lhs.width);
        
        for(size_t y = 0; y < lhs.height; y++) {
            for (size_t x = 0; x < lhs.width; x++) {
                m.at(y, x) = lhs.at(y, x) + rhs.at(y, x);
            }
        }

        return m;
    }

    friend Matrix operator-(const Matrix& lhs, const Matrix& rhs) {
        if (lhs.width != rhs.width || lhs.height != rhs.height) {
            std::cout << "sizes dont match for -. sizes " << "(" << lhs.height << ", " << lhs.width << ")" << " (" << rhs.height << ", " << rhs.width << ")" << std::endl;
        }

        Matrix m = Matrix::zeros(lhs.height, lhs.width);
        
        for(size_t y = 0; y < lhs.height; y++) {
            for (size_t x = 0; x < lhs.width; x++) {
                m.at(y, x) = lhs.at(y, x) - rhs.at(y, x);
            }
        }

        return m;
    }

    Matrix& operator-() {
        for (size_t i = 0; i < width * height; i++) {
            data[i] = -data[i];
        }
        return *this;
    }

    static Matrix hadamard_product(const Matrix& lhs, const Matrix& rhs) {
        Matrix m(lhs.height, lhs.width);

        for(size_t y = 0; y < lhs.height; y++) {
            for (size_t x = 0; x < lhs.width; x++) {
                m.at(y, x) = lhs.at(y, x) * rhs.at(y, x);
            }
        }

        return m;
    }

    Matrix& transpose() {
        std::vector<float> old_data = data;

        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                data[x * height + y] = old_data[y * width + x];
            }
        }

        size_t tmp = width;
        width = height;
        height = tmp;

        return *this;
    }

    Matrix transposed() {
        Matrix m = *this;
        return m.transpose();
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& rhs) {
        for (size_t y = 0; y < rhs.height; y++) {
            for (size_t x = 0; x < rhs.width; x++) {
                os << rhs.at(y, x) << " ";
            }
            os << std::endl;
        }

        return os;
    }

    std::string get_dim() const {
        return "(" + std::to_string(height) + ", " + std::to_string(width) + ")";
    }
};

// naive: 187ms

// int main() {
//     Matrix x = Matrix::from_random(2, 2);
//     std::cout << x << std::endl;
//     std::cout << x.transposed() << std::endl;

//     Matrix a = Matrix::from_random(1028, 1028);
//     Matrix b = Matrix::from_random(1028, 1028);

//     long long s = 0;

//     for (int i = 0; i < 10; i++) {

//         auto start = std::chrono::steady_clock::now();
//         a * b;

//         auto end = std::chrono::steady_clock::now();
//         std::chrono::duration<double, std::milli> ms = end - start;
//         s += ms.count();
//     }

//     std::cout << s / 10.0 << "ms" << std::endl;

//     // std::cout << a * b << std::endl;
// }

/*
    friend Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
        if (lhs.width != rhs.height) {
            std::cout << "width or height not equal in mat mult" << std::endl;
            int a = 1 / 0;
        }

        Matrix m = Matrix::zeros(lhs.height, rhs.width);

        for (size_t y = 0; y < lhs.height; y++) {
            for (size_t x = 0; x < rhs.width; x++) {
                float s = 0;
                for (size_t i = 0; i < lhs.width; i++) {
                    s += lhs.at(y, i) * rhs.at(i, x);
                }
                m.at(y, x) = s;
            }
        }

        return m;
    }
*/