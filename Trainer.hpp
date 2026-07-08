#pragma once

#include "FFNN.hpp"
#include "DataSet.hpp"

#include <string>
#include <random>
#include <limits>
#include <iostream>

namespace nn_utils {

// fraction of samples where the highest-value output row matches the highest-value label row
inline float argmax_accuracy(const FFNN& ffnn, const DataSet& data) {
    int total_right = 0;
    const Eigen::MatrixXf predictions = ffnn.forward(data.inputs);

    for (Eigen::Index i = 0; i < data.inputs.cols(); i++) {
        Eigen::Index answer = 0, answer_col = 0;
        data.labels.col(i).maxCoeff(&answer, &answer_col);

        Eigen::Index prediction = 0, prediction_col = 0;
        predictions.col(i).maxCoeff(&prediction, &prediction_col);

        if (answer == prediction) total_right++;
    }

    return static_cast<float>(total_right) / static_cast<float>(data.size());
}

}

struct TrainSettings {
    size_t num_steps;
    size_t batch_size;
    size_t eval_interval = 1000;
    size_t print_interval = 100;
    unsigned int seed = std::random_device{}();
    std::string checkpoint_path = "";   // empty = no checkpointing on new best
};

struct TrainResult {
    float best_score;
    size_t best_step;
    float final_cost;
};

/*
runs a training loop: draws random minibatches, steps the optimiser, and periodically evaluates
against eval (higher is better), tracking the best score seen and writing a checkpoint each time
it improves (if checkpoint_path is set)
*/
template<typename Optimiser, typename EvalFn>
TrainResult train(FFNN& ffnn, Optimiser& opt, const DataSet& train_data, const TrainSettings& settings, EvalFn eval) {
    Batcher batcher = Batcher::from_dataset(train_data, settings.seed);

    Eigen::MatrixXf inputs;
    Eigen::MatrixXf targets;

    float best_score = std::numeric_limits<float>::lowest();
    size_t best_step = 0;
    float final_cost = 0.0f;

    for (size_t step = 0; step < settings.num_steps; step++) {
        batcher.next_batch(settings.batch_size, inputs, targets);
        final_cost = opt.step(inputs, targets);

        if (step % settings.eval_interval == 0) {
            const float score = eval(ffnn);
            if (score > best_score) {
                best_score = score;
                best_step = step;
                if (!settings.checkpoint_path.empty()) {
                    ffnn.write_to_file(settings.checkpoint_path);
                }
            }
        }

        if (step % settings.print_interval == 0) {
            std::cout << "step " << step + 1 << "/" << settings.num_steps << " | cost " << final_cost << " | best score " << best_score << " at " << best_step << "\n";
        }
    }

    return TrainResult{best_score, best_step, final_cost};
}

// same as train(), but on_batch(inputs, targets) runs on every minibatch before the optimiser
// step, e.g. for input augmentation
template<typename Optimiser, typename EvalFn, typename OnBatch>
TrainResult train(FFNN& ffnn, Optimiser& opt, const DataSet& train_data, const TrainSettings& settings, EvalFn eval, OnBatch on_batch) {
    Batcher batcher = Batcher::from_dataset(train_data, settings.seed);

    Eigen::MatrixXf inputs;
    Eigen::MatrixXf targets;

    float best_score = std::numeric_limits<float>::lowest();
    size_t best_step = 0;
    float final_cost = 0.0f;

    for (size_t step = 0; step < settings.num_steps; step++) {
        batcher.next_batch(settings.batch_size, inputs, targets);
        on_batch(inputs, targets);
        final_cost = opt.step(inputs, targets);

        if (step % settings.eval_interval == 0) {
            const float score = eval(ffnn);
            if (score > best_score) {
                best_score = score;
                best_step = step;
                if (!settings.checkpoint_path.empty()) {
                    ffnn.write_to_file(settings.checkpoint_path);
                }
            }
        }

        if (step % settings.print_interval == 0) {
            std::cout << "step " << step + 1 << "/" << settings.num_steps << " | cost " << final_cost << " | best score " << best_score << " at " << best_step << "\n";
        }
    }

    return TrainResult{best_score, best_step, final_cost};
}

// same as train(), but steps an lr scheduler (anything with .step() and a .lr member) at every
// eval interval and applies its rate to the optimiser
template<typename Optimiser, typename Scheduler, typename EvalFn>
TrainResult train_with_scheduler(FFNN& ffnn, Optimiser& opt, Scheduler& scheduler, const DataSet& train_data, const TrainSettings& settings, EvalFn eval) {
    Batcher batcher = Batcher::from_dataset(train_data, settings.seed);

    Eigen::MatrixXf inputs;
    Eigen::MatrixXf targets;

    float best_score = std::numeric_limits<float>::lowest();
    size_t best_step = 0;
    float final_cost = 0.0f;

    for (size_t step = 0; step < settings.num_steps; step++) {
        batcher.next_batch(settings.batch_size, inputs, targets);
        final_cost = opt.step(inputs, targets);

        if (step % settings.eval_interval == 0) {
            const float score = eval(ffnn);
            if (score > best_score) {
                best_score = score;
                best_step = step;
                if (!settings.checkpoint_path.empty()) {
                    ffnn.write_to_file(settings.checkpoint_path);
                }
            }

            scheduler.step();
            opt.lr = scheduler.lr;
        }

        if (step % settings.print_interval == 0) {
            std::cout << "step " << step + 1 << "/" << settings.num_steps << " | cost " << final_cost << " | lr " << opt.lr << " | best score " << best_score << " at " << best_step << "\n";
        }
    }

    return TrainResult{best_score, best_step, final_cost};
}
