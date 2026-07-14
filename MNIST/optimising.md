# Hyperparameter history

Model names (ascending): Cirrus → Cumulus → Nimbus → Cumulonimbus
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
