#pragma once

#include "FFNN.hpp"
#include "DataSet.hpp"

#include <string>
#include <random>
#include <limits>
#include <iostream>
#include <type_traits>
#include <iomanip>

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

namespace nn_utils {

// no-op default for the on_batch hook in train() / train_with_scheduler()
inline void no_op_on_batch(Eigen::MatrixXf&, Eigen::MatrixXf&) {}

// tag type selecting the no-scheduler path in train_loop()
struct NoScheduler {};

/*
shared training loop: draws random minibatches, runs on_batch(inputs, targets) before the
optimiser step (e.g. for input augmentation), steps the optimiser, and periodically evaluates
against eval (higher is better), tracking the best score seen and writing a checkpoint each
time it improves (if checkpoint_path is set). unless scheduler is a NoScheduler, it is also
stepped each iteration (anything with .step() and a .lr member) and its rate applied to the
optimiser
*/
template<typename Optimiser, typename Scheduler, typename EvalFn, typename OnBatch>
TrainResult train_loop(FFNN& ffnn, Optimiser& opt, Scheduler& scheduler, const DataSet& train_data, const TrainSettings& settings, EvalFn eval, OnBatch on_batch) {
    constexpr bool has_scheduler = !std::is_same_v<Scheduler, NoScheduler>;

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

        if constexpr (has_scheduler) {
            scheduler.step();
            opt.lr = scheduler.lr;
        }

        if (step % settings.print_interval == 0) {
            std::cout << "step " << std::setw(static_cast<int>(std::log10(settings.num_steps)) + 1) << step + 1 << "/" << settings.num_steps << " | cost " << std::fixed << std::setprecision(6) << final_cost;
            if constexpr (has_scheduler) std::cout << " | lr " << opt.lr;
            std::cout << " | best score " << std::fixed << std::setprecision(6) << best_score << " at " << best_step << "\n";
        }
    }

    return TrainResult{best_score, best_step, final_cost};
}

}

// runs a training loop, tracking the best eval score and checkpointing on improvement
template<typename Optimiser, typename EvalFn>
TrainResult train(FFNN& ffnn, Optimiser& opt, const DataSet& train_data, const TrainSettings& settings, EvalFn eval) {
    return train(ffnn, opt, train_data, settings, eval, nn_utils::no_op_on_batch);
}

// same as train(), but on_batch(inputs, targets) runs on every minibatch before the optimiser
// step, e.g. for input augmentation
template<typename Optimiser, typename EvalFn, typename OnBatch>
TrainResult train(FFNN& ffnn, Optimiser& opt, const DataSet& train_data, const TrainSettings& settings, EvalFn eval, OnBatch on_batch) {
    nn_utils::NoScheduler no_scheduler;
    return nn_utils::train_loop(ffnn, opt, no_scheduler, train_data, settings, eval, on_batch);
}

// same as train(), but steps an lr scheduler (anything with .step() and a .lr member) at every
// eval interval and applies its rate to the optimiser
template<typename Optimiser, typename Scheduler, typename EvalFn>
TrainResult train_with_scheduler(FFNN& ffnn, Optimiser& opt, Scheduler& scheduler, const DataSet& train_data, const TrainSettings& settings, EvalFn eval) {
    return train_with_scheduler(ffnn, opt, scheduler, train_data, settings, eval, nn_utils::no_op_on_batch);
}

// same as train_with_scheduler(), but on_batch(inputs, targets) runs on every minibatch before
// the optimiser step, e.g. for input augmentation
template<typename Optimiser, typename Scheduler, typename EvalFn, typename OnBatch>
TrainResult train_with_scheduler(FFNN& ffnn, Optimiser& opt, Scheduler& scheduler, const DataSet& train_data, const TrainSettings& settings, EvalFn eval, OnBatch on_batch) {
    return nn_utils::train_loop(ffnn, opt, scheduler, train_data, settings, eval, on_batch);
}
