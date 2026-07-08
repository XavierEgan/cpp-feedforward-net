// clang++ -std=c++23 -O0 -I.. tests/test_gradients.cpp -o test_gradients && ./test_gradients

#include "../FFNN.hpp"
#include "../Workspace.hpp"
#include "../NN_Utils.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

// numerically differentiates the total cost (data cost + regularization penalty) with respect
// to a single weight/bias entry, matching the penalty terms nn_utils::add_regularization adds
float central_diff_cost(FFNN& ffnn, size_t l, bool is_weight, int r, int c, const Eigen::MatrixXf& input, const Eigen::MatrixXf& target, CostType cost_type, RegularizationType reg_type, float reg_lambda, float h) {
    Eigen::MatrixXf& param = is_weight ? ffnn.get_weight(l) : ffnn.get_bias(l);
    const float orig = param(r, c);

    auto total_cost = [&]() -> float {
        const float data_cost = nn_utils::cost(ffnn.forward(input), target, cost_type);
        if (!is_weight) return data_cost;

        const Eigen::MatrixXf& w = ffnn.get_weight(l);
        if (reg_type == RegularizationType::l2) return data_cost + reg_lambda * 0.5f * w.squaredNorm();
        if (reg_type == RegularizationType::l1) return data_cost + reg_lambda * w.cwiseAbs().sum();
        return data_cost;
    };

    param(r, c) = orig + h;
    const float cost_plus = total_cost();

    param(r, c) = orig - h;
    const float cost_minus = total_cost();

    param(r, c) = orig;
    return (cost_plus - cost_minus) / (2.0f * h);
}

void check(const std::string& name, std::vector<size_t> shape, std::vector<ActivationFunc> funcs, CostType cost_type, RegularizationType reg_type = RegularizationType::none) {
    FFNN ffnn = FFNN::from_random_he_scaling(shape, funcs);
    BackpropWorkspace ws = BackpropWorkspace::from_shape(shape);

    Eigen::MatrixXf input = Eigen::MatrixXf::Random(shape.front(), 5);
    Eigen::MatrixXf target;
    if (cost_type == CostType::categorical_cross_entropy) {
        target = Eigen::MatrixXf::Zero(shape.back(), 5);
        for (int col = 0; col < 5; col++) target(col % shape.back(), col) = 1.0f;
    } else if (cost_type == CostType::binary_cross_entropy) {
        target = (Eigen::MatrixXf::Random(shape.back(), 5).array() > 0).cast<float>();
    } else {
        target = Eigen::MatrixXf::Random(shape.back(), 5);
    }

    ffnn.backward(input, target, cost_type, ws);

    const float reg_lambda = 1e-2f;
    float max_rel_err = 0.0f;
    const float h = 1e-2f;
    // below this magnitude, float32 central-difference rounding noise (~epsilon / h) dominates
    // the true gradient, so relative error is meaningless and the entry is skipped
    const float denom_floor = 1e-3f;

    for (size_t l = 1; l < ffnn.depth(); l++) {
        Eigen::MatrixXf& w = ffnn.get_weight(l);
        Eigen::MatrixXf analytic_w = ws.weight_grad.at(l - 1);
        nn_utils::add_regularization(analytic_w, w, reg_type, reg_lambda);

        // relu has a kink at z == 0: skip units whose pre-activation lands near it for any
        // sample in the batch, since a finite-difference step could cross the kink
        const bool is_relu = funcs.at(l - 1) == ActivationFunc::relu;
        const Eigen::MatrixXf& z_layer = ws.fwd.z.at(l - 1);

        for (int r = 0; r < w.rows(); r++) {
            if (is_relu && (z_layer.row(r).cwiseAbs().array() < 0.05f).any()) continue;

            for (int c = 0; c < w.cols(); c++) {
                // l1's sign(w) term is discontinuous at w == 0, same problem as the relu kink
                if (reg_type == RegularizationType::l1 && std::abs(w(r, c)) < h) continue;

                const float analytic = analytic_w(r, c);
                const float numeric = central_diff_cost(ffnn, l, true, r, c, input, target, cost_type, reg_type, reg_lambda, h);
                const float denom = std::max(std::abs(analytic), std::abs(numeric));
                if (denom < denom_floor) continue;

                max_rel_err = std::max(max_rel_err, std::abs(analytic - numeric) / denom);
            }
        }

        Eigen::MatrixXf& b = ffnn.get_bias(l);
        for (int r = 0; r < b.rows(); r++) {
            if (is_relu && (z_layer.row(r).cwiseAbs().array() < 0.05f).any()) continue;

            const float analytic = ws.bias_grad.at(l - 1)(r, 0);
            const float numeric = central_diff_cost(ffnn, l, false, r, 0, input, target, cost_type, reg_type, reg_lambda, h);
            const float denom = std::max(std::abs(analytic), std::abs(numeric));
            if (denom < denom_floor) continue;

            max_rel_err = std::max(max_rel_err, std::abs(analytic - numeric) / denom);
        }
    }

    std::cout << name << ": max relative error = " << max_rel_err << (max_rel_err < 1e-2f ? "  PASS" : "  FAIL") << "\n";
    assert(max_rel_err < 1e-2f);
}

int main() {
    srand(1);   // deterministic weight/input initialization for reproducible results

    check("mse + tanh", {3, 4, 2}, {ActivationFunc::tan_h, ActivationFunc::tan_h}, CostType::mse);
    check("mse + sigmoid", {3, 4, 2}, {ActivationFunc::tan_h, ActivationFunc::sigmoid}, CostType::mse);
    check("mse + linear", {3, 4, 2}, {ActivationFunc::tan_h, ActivationFunc::linear}, CostType::mse);
    check("mse + relu", {3, 5, 2}, {ActivationFunc::relu, ActivationFunc::linear}, CostType::mse);
    check("bce + sigmoid", {3, 4, 2}, {ActivationFunc::tan_h, ActivationFunc::sigmoid}, CostType::binary_cross_entropy);
    check("cce + softmax", {3, 4, 3}, {ActivationFunc::tan_h, ActivationFunc::softmax}, CostType::categorical_cross_entropy);

    check("mse + tanh + l2", {3, 4, 2}, {ActivationFunc::tan_h, ActivationFunc::tan_h}, CostType::mse, RegularizationType::l2);
    check("mse + tanh + l1", {3, 4, 2}, {ActivationFunc::tan_h, ActivationFunc::tan_h}, CostType::mse, RegularizationType::l1);
    check("cce + softmax + l2", {3, 4, 3}, {ActivationFunc::tan_h, ActivationFunc::softmax}, CostType::categorical_cross_entropy, RegularizationType::l2);
    check("cce + softmax + l1", {3, 4, 3}, {ActivationFunc::tan_h, ActivationFunc::softmax}, CostType::categorical_cross_entropy, RegularizationType::l1);

    std::cout << "ALL PASS\n";
    return 0;
}
