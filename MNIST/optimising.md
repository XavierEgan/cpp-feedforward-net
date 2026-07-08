

# Profiling
We see from this Flame graph that the runtime is dominated by += and get_random_batch()
![alt text](image.png).

First lets investigate the +=. It's likely from the noise injection:
```c++
// for each epoch
for (Eigen::Index j = 0; j < inputs.cols(); j++) {
    if (static_cast<double>(rand()) / RAND_MAX < settings.chance_for_noise) {
        auto a = (Eigen::MatrixXf::Random(inputs.rows(), 1).array() + 1).matrix() * (settings.noise / 2.0f);
        inputs.col(j) += a;
    }
}
```
First lets confirm its the += by commenting it out and checking the flame graph:

# Hyperparameter history

99.04% at 159000 steps
```
Settings settings{
        .ffnn_shape = {IMG_SIZE, 1024, 512, 255, 128, 64, 32, 16, NUM_CLASSES},
        .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax},
        .cost_type = CostType::categorical_cross_entropy,
        .reg_type = RegularizationType::none,
        .num_steps = 1000000,
        .seed = 1,
        .batch_size = 1000,
        .lr = 0.0005,
        .noise = 1,
        .chance_for_noise = 0.5
    };
```

98.88%
```
Settings settings{
        .ffnn_shape = {IMG_SIZE, 1024, 512, 255, 128, 64, 32, 16, NUM_CLASSES},
        .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::sigmoid},
        .cost_type = CostType::binary_cross_entropy,
        .reg_type = RegularizationType::none,
        .num_steps = 10000,
        .seed = 1,
        .batch_size = 1000,
        .lr = 0.0005,
        .noise = 1,
        .chance_for_noise = 0.5
    };
```

98.09% - went to 5k generations
```
Settings settings{
    .ffnn_shape = {IMG_SIZE, 512, 255, 128, 64, 32, 16, NUM_CLASSES},
    .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::sigmoid},
    .cost_type = CostType::binary_cross_entropy,
    .reg_type = RegularizationType::none,
    .num_steps = 10000,
    .seed = 1,
    .batch_size = 1000,
    .lr = 0.0005
};
```
