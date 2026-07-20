#include "../DataSet.hpp"
#include "../AdamOptimiser.hpp"
#include "../Trainer.hpp"
#include "../LrSchedulers.hpp"

#include <string>
#include <filesystem>
#include <random>
#include <iostream>
#include <vector>

constexpr int NOISE_SIZE = 10;
constexpr int IMG_SIZE = 784;

constexpr float INPUT_NOISE_STDDEV = 0.1f;

// weight on the generator's "look real" (jibberish) gradient relative to its "look like the
// requested digit" (adversary) gradient. both are summed into the generator's output delta.
constexpr float REALNESS_WEIGHT = 100f;

void print_image(const Eigen::MatrixXf& image) {
    for (Eigen::Index i = 0; i < 28; i++) {
        for (Eigen::Index j = 0; j < 28; j++) {
            std::cout << (image(i * 28 + j, 0) > 0.5f ? "██" : "░░");
        }
        std::cout << "\n";
    }
}

// each optimiser holds an FFNN& to the network it trains, so the FFNNs must outlive the
// optimisers and never be relocated once the optimisers reference them. Context therefore owns
// the FFNNs by value and wires each optimiser to its own member in the constructor, and is made
// non-copyable/non-movable so those references can never be left dangling by a copy or move.
struct Context {
    // generates the image from a one-hot vector of the digit to generate
    FFNN image_gen;

    // predicts the digit from an image, used to compute the loss for the generator
    FFNN adversary;

    // predicts if a given image is a digit or jibberish, use to train the adversary
    FFNN jibberish;

    // each optimiser owns the BackpropWorkspace it reads/writes during a step, so forward passes
    // meant to feed a step_from_output_delta() must be written into that optimiser's .ws.fwd
    AdamOptimiser image_gen_optimiser;
    AdamOptimiser adversary_optimiser;
    AdamOptimiser jibberish_optimiser;

    Context(FFNN image_gen, FFNN adversary, FFNN jibberish)
        : image_gen(std::move(image_gen)), adversary(std::move(adversary)), jibberish(std::move(jibberish)),
          image_gen_optimiser(AdamOptimiser::from_ffnn(this->image_gen, CostType::categorical_cross_entropy)),
          adversary_optimiser(AdamOptimiser::from_ffnn(this->adversary, CostType::categorical_cross_entropy)),
          jibberish_optimiser(AdamOptimiser::from_ffnn(this->jibberish, CostType::binary_cross_entropy)) {}

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;
};

Context get_context() {
    return Context(
        FFNN::from_random_he_scaling(
            {NOISE_SIZE, 512, 1024, IMG_SIZE},
            {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::linear}
        ),
        FFNN::from_file("MNIST/models/MNIST-Cirrus-1.dat"),
        FFNN::from_random_he_scaling(
            {IMG_SIZE, 1024, 512, 1},
            {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::sigmoid}
        )
    );
}

DataSet get_jibberish_training_data(int seed = 1) {
    // input: image, output: 1 if legable digit, 0 if jibberish/noise/whatever
    // MNIST is positive examples, random noise for negative examples

    // load positive examples from MNIST train and test sets
    DataSet positive_examples = DataSet::from_file("MNIST/bin/train.dat");
    positive_examples.append(DataSet::from_file("MNIST/bin/test.dat"));

    positive_examples.labels = Eigen::MatrixXf::Ones(1, positive_examples.size());

    // generate negative examples from random noise
    DataSet negative_examples = DataSet::empty(IMG_SIZE, 1);

    Eigen::MatrixXf noise(IMG_SIZE, positive_examples.size());

    std::mt19937 rng(seed);
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);
    for (Eigen::Index i = 0; i < positive_examples.size(); i++) {
        for (Eigen::Index j = 0; j < noise.rows(); j++) {
            noise(j, i) = noise_dist(rng);
        }
    }
    negative_examples.inputs = noise;
    negative_examples.labels = Eigen::MatrixXf::Zero(1, negative_examples.size());

    DataSet training_data = positive_examples;
    training_data.append(negative_examples);
    return training_data;
}

// real MNIST digits labelled positive (1), used as the "real" examples the jibberish
// discriminator is co-trained on each step of the GAN loop against generated negatives
DataSet get_real_digits() {
    DataSet digits = DataSet::from_file("MNIST/bin/train.dat");
    digits.append(DataSet::from_file("MNIST/bin/test.dat"));
    digits.labels = Eigen::MatrixXf::Ones(1, digits.size());
    return digits;
}

void train_jibberish_model(Context& ctx, size_t batch_size, size_t num_steps, int seed = 1) {
    DataSet training_data = get_jibberish_training_data(seed);

    std::cout << "training jibberish model on " << training_data.size() << " samples\n";

    TrainSettings settings {
        .num_steps = num_steps,
        .batch_size = batch_size,
        .eval_interval = 100,   // else eval only runs at step 0 (on the untrained model) and "best score" stays ~0.5
        .seed = static_cast<unsigned int>(seed)
    };

    // fraction of samples where (prediction > 0.5) matches the binary label
    auto eval = [&](const FFNN& ffnn) {
        const Eigen::MatrixXf predictions = ffnn.forward(training_data.inputs);
        const Eigen::Array<bool, 1, Eigen::Dynamic> predicted_positive = predictions.array().row(0) > 0.5f;
        const Eigen::Array<bool, 1, Eigen::Dynamic> label_positive = training_data.labels.array().row(0) > 0.5f;
        const Eigen::Index correct = (predicted_positive == label_positive).count();
        return static_cast<float>(correct) / static_cast<float>(training_data.size());
    };

    train(ctx.jibberish, ctx.jibberish_optimiser, training_data, settings, eval);
}

/*
3-network conditional GAN:
  - image_gen  (generator):     digit one-hot -> image
  - adversary  (FROZEN):        image -> which digit? (conditioning: "look like the requested digit")
  - jibberish  (discriminator): image -> real or fake? (co-trained: "look like a real digit at all")

0) pre-train jibberish model so it starts as a competent real/noise discriminator

per step:
1) generate a batch of images from requested digits
2) UPDATE GENERATOR with two summed gradients w.r.t. the generated images:
   a) identity: backprop the frozen adversary's digit-classification loss (target = requested digit)
   b) realness: backprop the jibberish discriminator's loss for target "real" (=1)
3) UPDATE DISCRIMINATOR (jibberish): train on real MNIST digits (label 1) and this step's
   generated images (label 0). the adversary is never updated - it stays a fixed digit classifier.
*/

int main() {
    // one-hot vector for each digit, used both as the generator's input and the adversary's
    // target - digit_to_input.at(d) is what "asking for a d" looks like
    std::vector<Eigen::MatrixXf> digit_to_input(NOISE_SIZE);
    for (int digit = 0; digit < NOISE_SIZE; digit++) {
        Eigen::MatrixXf input = Eigen::MatrixXf::Zero(NOISE_SIZE, 1);
        input.row(digit).setOnes();
        digit_to_input.at(digit) = input;
    }

    constexpr size_t jibberish_steps = 1000;
    constexpr size_t num_steps = 10000;
    constexpr size_t batch_size = 128;

    Context ctx = get_context();

    // 0) pre-train the jibberish discriminator so it starts competent (real digits vs noise)
    train_jibberish_model(ctx, batch_size, jibberish_steps);

    // real MNIST digits (label 1) - the discriminator's positive examples, sampled fresh each
    // step to sit against the generator's outputs (label 0)
    const DataSet real_digits = get_real_digits();
    Batcher real_batcher = Batcher::from_dataset(real_digits, 1);

    std::mt19937 rng(1);
    std::normal_distribution<float> noise_dist(0.0f, INPUT_NOISE_STDDEV);
    std::uniform_int_distribution<int> digit_dist(0, NOISE_SIZE - 1);

    Eigen::MatrixXf target(NOISE_SIZE, batch_size); // the digit we ask the generator for (and the adversary's target)
    Eigen::MatrixXf input(NOISE_SIZE, batch_size);  // input to the image generator

    // the discriminator's "fake" label for generated images (real MNIST batches carry label 1
    // already, from get_real_digits). the generator's realness target "real" (=1) is real_ones.
    const Eigen::MatrixXf real_ones = Eigen::MatrixXf::Ones(1, batch_size);
    const Eigen::MatrixXf fake_labels = Eigen::MatrixXf::Zero(1, batch_size);

    // reusable scratch for the generator's realness gradient (backprop through the discriminator
    // without updating it, so it can't use its own optimiser's step)
    BackpropWorkspace jibberish_grad_ws = BackpropWorkspace::from_shape(ctx.jibberish.network_shape);

    Eigen::MatrixXf real_batch_inputs;
    Eigen::MatrixXf real_batch_labels;

    for (size_t step = 0; step < num_steps; step++) {
        // MAIN LOOP:
        //     MAKE a batch of one hot vectors with slight noise added
        for (size_t i = 0; i < batch_size; i++) {
            const int digit = digit_dist(rng);
            input.col(i) = digit_to_input.at(digit);

            for (Eigen::Index j = 0; j < input.rows(); j++) {
                input(j, i) += noise_dist(rng);
            }

            target.col(i) = digit_to_input.at(digit);
        }

        //     FORWARD through the image generator (generates a bunch of images). this forward
        //     pass must land in the generator optimiser's own workspace, since the generator's
        //     step_from_output_delta() below backprops through that same workspace
        ctx.image_gen.forward(input, ctx.image_gen_optimiser.ws.fwd);

        const Eigen::MatrixXf generated_images = ctx.image_gen_optimiser.ws.fwd.a.back();

        // ===== UPDATE GENERATOR: sum of two gradients w.r.t. the generated images =====

        // (a) IDENTITY gradient: backprop the frozen adversary's classification loss (target =
        //     the requested digit). the adversary is not updated - we only want dLoss/d(images).
        BackpropWorkspace& adversary_ws = ctx.adversary_optimiser.ws;
        ctx.adversary.forward(generated_images, adversary_ws.fwd);
        ctx.adversary.fill_output_delta(target, CostType::categorical_cross_entropy, adversary_ws);
        ctx.adversary.backward(generated_images, adversary_ws);
        const Eigen::MatrixXf grad_identity = ctx.adversary.grad_wrt_input(adversary_ws);

        // (b) REALNESS gradient: backprop the jibberish discriminator's loss for target "real" (=1).
        //     with sigmoid + binary_cross_entropy the output delta simplifies to (output - target),
        //     so pushing towards "real" is (sigmoid_out - 1). again the discriminator is NOT updated
        //     here (scratch workspace), we only want dLoss/d(images).
        const Eigen::MatrixXf& jibberish_out = ctx.jibberish.forward(generated_images, jibberish_grad_ws.fwd); // 1 x batch
        jibberish_grad_ws.delta.at(ctx.jibberish.depth() - 2) = jibberish_out - real_ones; // (sigmoid_out - 1)
        ctx.jibberish.backward(generated_images, jibberish_grad_ws);
        const Eigen::MatrixXf grad_realness = ctx.jibberish.grad_wrt_input(jibberish_grad_ws);

        //     the generator's output layer is linear, so dLoss/dz == dLoss/d(generated_images);
        //     sum the two gradients and take one optimiser step on the generator
        const Eigen::MatrixXf gen_output_delta = grad_identity + REALNESS_WEIGHT * grad_realness;
        ctx.image_gen_optimiser.step_from_output_delta(input, gen_output_delta);

        // ===== UPDATE DISCRIMINATOR (jibberish): real MNIST (label 1) vs generated (label 0) =====
        //     labels are fixed by provenance, no per-image judgement. training on real digits keeps
        //     it from collapsing to "everything is fake"; training on generated negatives keeps its
        //     realness gradient (used above) honest against what the generator actually produces.
        real_batcher.next_batch(batch_size, real_batch_inputs, real_batch_labels); // labels are all 1 (real)
        ctx.jibberish_optimiser.step(real_batch_inputs, real_batch_labels);
        ctx.jibberish_optimiser.step(generated_images, fake_labels);
    }

    for (int digit = 0; digit < NOISE_SIZE; digit++) {
        std::cout << "digit " << digit << ":\n";
        print_image(ctx.image_gen.forward(digit_to_input.at(digit)));
    }

    ctx.image_gen.write_to_file("image-gen/generator.dat");
}