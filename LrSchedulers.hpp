#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace nn_utils {
constexpr double lr_eps = 1e-200;

struct LRSchedulerLinear {
    double step_size;
    double lr;
    double lr_min;

    static LRSchedulerLinear from_num_generation(double lr_initial, double lr_final, int max_generations) {
        double step = (static_cast<double>(lr_initial) - static_cast<double>(lr_final)) / static_cast<double>(max_generations);

        return LRSchedulerLinear(step, lr_initial, lr_final);
    }

    void step() {
        lr = std::max(lr - step_size, lr_min);
    }
    bool is_done() {
        return lr - step_size <= lr_min;
    }


private:
    LRSchedulerLinear(double step, double lr, double lr_min) : step_size(step), lr(lr), lr_min(lr_min) {}
};

struct LRSchedulerExponential {
    double step_size;
    double lr;
    double lr_min;

    static LRSchedulerExponential from_num_generation(double lr_initial, double lr_final, int max_generations) {
        if (max_generations < 1) {
            throw std::invalid_argument("max_generations cant be 0");
        }
        lr_initial = std::max(lr_initial, lr_eps);

        double step_size = pow(static_cast<double>(lr_final) / static_cast<double>(lr_initial), 1.0 / static_cast<double>(max_generations));

        return LRSchedulerExponential(step_size, lr_initial, lr_final);
    }

    void step() {
        lr = std::max(lr * step_size, lr_min);
    }

    bool is_done() {
        return lr * step_size <= lr_min;
    }

private:
    LRSchedulerExponential(double step_size, double lr, double lr_min) : step_size(step_size), lr(lr), lr_min(lr_min) {}
};

struct LRSchedulerDecayOnPlateau {
    double step_size;
    double lr;
    double lr_min;

    int patience;
    int gens_since_improvement;
    double best_error = std::numeric_limits<double>::max();

    static LRSchedulerDecayOnPlateau from_num_generation(double lr_initial, double lr_final, int min_generations, int patience) {
        if (min_generations < 1) {
            throw std::invalid_argument("min_generations cant be 0");
        }
        lr_initial = std::max(lr_initial, lr_eps);

        min_generations *= patience;

        /*
        L_f = L_i * (S_s)^(G_m / P)
        therfore
        S_s = (L_f / L_i) ^ (P / G_m)
        */

        double step_size = pow(lr_final / lr_initial, static_cast<double>(patience) / static_cast<double>(min_generations));

        return LRSchedulerDecayOnPlateau(step_size, lr_initial, lr_final, patience);
    }

    void step(double error) {
        if (error < best_error) {
            best_error = error;
            gens_since_improvement = 0;
            return;
        }

        gens_since_improvement++;
        if (gens_since_improvement >= patience) {
            gens_since_improvement = 0;
            lr = std::max(lr * step_size, lr_min);
        }
    }

    bool is_done() {
        return lr * step_size <= lr_min;
    }

private:
    LRSchedulerDecayOnPlateau(double step_size, double lr, double lr_min, int patience) : step_size(step_size), lr(lr), lr_min(lr_min), patience(patience), gens_since_improvement(0) {}
};
}