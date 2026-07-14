# Hyperparameter history

Model names (ascending): Cirrus → Cumulus → Nimbus → Cumulonimbus

Along the way, found and fixed a bug in MNIST.cpp's LRSchedulerLinear call: args were passed as
(lr_min, lr_initial) instead of (lr_initial, lr_min), so lr was ramping up over training instead
of decaying. All four tiers below were tuned/trained after the fix.

For every tier here, noise augmentation and L2 reg both hurt slightly or did nothing - these
networks don't have enough capacity relative to the ~60k training set to overfit, so the
regularization is pure signal loss. chance_for_noise = 0.0 and reg_type = none throughout.
Also true across all tiers: test accuracy plateaus well before num_steps ends (bigger nets
plateau later) - training cost keeps dropping to ~0 after that point, i.e. they're fitting
train-batch noise, not learning anything that moves test accuracy. num_steps below has a bit
of margin past the observed plateau, not a tight cutoff.

## Final tiers (checkpoints in MNIST/models/)

**Cirrus** - no hidden layer (plain softmax regression, 7850 params) - **92.92%**, plateaus ~step 3400
regardless of num_steps (tried up to 20k) - this is the accuracy ceiling for a linear model on raw pixels.
```
Settings settings{
    .ffnn_shape = {IMG_SIZE, NUM_CLASSES},
    .ffnn_funcs = {ActivationFunc::softmax},
    .cost_type = CostType::categorical_cross_entropy,
    .reg_type = RegularizationType::none,
    .num_steps = 5000,
    .seed = 1,
    .batch_size = 1000,
    .lr = 0.005,
    .noise = 1,
    .chance_for_noise = 0.0
};
```

**Cumulus** - one 128-wide hidden layer (~101k params) - **98.10%**, plateaus ~step 3300.
lr swept 0.0005-0.05 at short horizon; 0.02 was the sweet spot, higher started destabilizing.
```
Settings settings{
    .ffnn_shape = {IMG_SIZE, 128, NUM_CLASSES},
    .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::softmax},
    .cost_type = CostType::categorical_cross_entropy,
    .reg_type = RegularizationType::none,
    .num_steps = 10000,
    .seed = 1,
    .batch_size = 1000,
    .lr = 0.02,
    .noise = 1,
    .chance_for_noise = 0.0
};
```

**Nimbus** - two hidden layers 512->256 (~537k params) - **98.60%**, plateaus ~step 4100.
Bigger net wants lower lr than Cumulus, as expected; 0.005 beat 0.002/0.008/0.01/0.02 at longer horizon.
```
Settings settings{
    .ffnn_shape = {IMG_SIZE, 512, 256, NUM_CLASSES},
    .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax},
    .cost_type = CostType::categorical_cross_entropy,
    .reg_type = RegularizationType::none,
    .num_steps = 10000,
    .seed = 1,
    .batch_size = 1000,
    .lr = 0.005,
    .noise = 1,
    .chance_for_noise = 0.0
};
```

**Cumulonimbus** - two hidden layers 2048->1024 (~4.6M params) - **98.68%**, plateaus ~step 8000.
Went looking for the actual ceiling here: tried a single 10000-wide layer (worse, 98.37% -
confirms depth matters more than raw width), and a deeper/wider 4096->2048->1024 funnel (also
worse, 98.65%, converges earlier then flattens, and much slower to train). 2048->1024 was the
best tradeoff found of the architectures tried. MNIST just doesn't have enough complexity to
reward going bigger than this - diminishing/negative returns past ~5M params.
```
Settings settings{
    .ffnn_shape = {IMG_SIZE, 2048, 1024, NUM_CLASSES},
    .ffnn_funcs = {ActivationFunc::relu, ActivationFunc::relu, ActivationFunc::softmax},
    .cost_type = CostType::categorical_cross_entropy,
    .reg_type = RegularizationType::none,
    .num_steps = 10000,
    .seed = 1,
    .batch_size = 1000,
    .lr = 0.003,
    .noise = 1,
    .chance_for_noise = 0.0
};
```

## Earlier exploratory runs

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
