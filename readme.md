# Cpp Feedforward Neural Net

A lightweight C++ feedforward neural network project built around `Eigen::MatrixXf`.

This repository is built for:

- learning and experimenting with core neural-network training loops,
- keeping the code easy to inspect and tweak,
- and providing a few concrete example projects (toy benchmark, spiral classifier, MNIST/Fashion-MNIST, tic-tac-toe).

The code is intentionally simple and header-driven, so the implementation is easy to read and modify.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Current Features](#current-features)
- [Repository Layout](#repository-layout)
- [Dependencies](#dependencies)
- [Build and Run](#build-and-run)
- [Quick Start API Example](#quick-start-api-example)
- [Included Example Projects](#included-example-projects)
- [Model and Dataset Files](#model-and-dataset-files)
- [Running the Tests](#running-the-tests)
- [Common Gotchas](#common-gotchas)
- [Performance Tips](#performance-tips)

---

## What This Project Does

At a high level, this repo provides a small neural-network training framework in modern C++.

You define:

- network shape (e.g. `{784, 256, 128, 10}`),
- activation functions per non-input layer,
- cost/loss function,
- optimizer,
- and training loop behavior.

The core `FFNN` type holds only the trained parameters (weights, biases, shape, activations).
Forward and backward passes are `const` and write into a caller-owned `Workspace`, so a single
`FFNN` can be evaluated repeatedly (or shared across threads) without reallocating.

Optimizers apply updates to weights/biases:

- plain gradient descent (`GradientDescentOptimiser`),
- Adam (`AdamOptimiser`).

`Trainer.hpp` ties a `DataSet`, an optimizer, and an eval function together into a training
loop with periodic evaluation, best-score tracking, and checkpointing.

---

## Current Features

### Network

- Dense feedforward architecture (`FFNN`), parameters only — no scratch state.
- He-style random initialization (`FFNN::from_random_he_scaling`).
- Versioned binary serialization (`write_to_file`, `from_file`).

### Activations

- `linear`
- `relu`
- `relu_clipped`
- `tan_h`
- `sigmoid`
- `softmax`

### Costs

- `mse`
- `binary_cross_entropy`
- `categorical_cross_entropy`

### Regularization

Set on the optimizer (`GradientDescentOptimiser`/`AdamOptimiser`), not on the model:

- `none`
- `l1`
- `l2`

### Optimizers

- `GradientDescentOptimiser::from_ffnn(...)`
- `AdamOptimiser::from_ffnn(...)`

### LR Schedulers

- linear decay
- exponential decay
- decay-on-plateau

`Trainer.hpp`'s `train_with_scheduler` steps any of these once per eval interval.

### Data

- `DataSet`: two big matrices (`input_dim x n`, `label_dim x n`), not one matrix per sample.
- `Batcher`: draws random mini-batches from a `DataSet`; its rng is seeded once at construction,
  so the same seed always reproduces the same sequence of batches.

---

## Repository Layout

Top-level highlights:

- `FFNN.hpp`: model parameters + forward/backward logic
- `Workspace.hpp`: `ForwardWorkspace`/`BackpropWorkspace` scratch buffers used by forward/backward
- `ActivationFunction.hpp`: activations + derivatives
- `CostType.hpp`: loss functions + derivatives
- `RegularizationType.hpp`: `none`/`l1`/`l2`
- `GradientDescentOptimiser.hpp`: SGD-like update step
- `AdamOptimiser.hpp`: Adam update step
- `LrSchedulers.hpp`: learning-rate scheduler helpers
- `NN_Utils.hpp`: regularization helper + small shape utilities
- `DataSet.hpp`: `DataSet` + `Batcher`
- `Trainer.hpp`: shared training loop + `nn_utils::argmax_accuracy`
- `main.cpp`: simple optimizer comparison demo
- `spiral/spiral.cpp`: self-contained spiral classifier demo (no data files needed)
- `MNIST/MNIST.cpp`: training example on the MNIST/Fashion-MNIST binary datasets
- `MNIST/Transform_Data_To_Binary.cpp`: converts the CSV datasets to the binary `DataSet` format
- `tictactoe/`: board engine, minimax agents, benchmarking utilities, and FFNN training experiments
- `tests/`: gradient-check, serialization, and `Batcher` tests
- `Eigen/`: vendored Eigen headers

---

## Dependencies

This project currently depends on:

- a C++23-capable compiler,
- Eigen headers (already included in this repo under `Eigen/`),
- CMake 3.20+ (optional — direct compiler invocations work too),
- optional OpenMP support for some training runs.

No external package manager is required for the core code path shown here.

---

## Build and Run

### CMake (recommended)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds `demo`, `mnist_train`, `mnist_transform`, `spiral_demo`, `ttt_interface`, and the
tests under `tests/`. Run `ctest --test-dir build` to run the tests.

Executables read relative paths (e.g. `MNIST/bin/test.dat`), so run them from the repo root:

```bash
./build/spiral_demo
```

`tictactoe/train.cpp` is an incomplete stub and is intentionally not built by CMake.

### Direct compiler invocations

Since this is a header-only project, everything also compiles with a single `g++`/`clang++`
command. Each example's own header comment has the exact command; the short version:

```bash
# basic optimizer demo
g++ -std=c++23 -O3 -o demo main.cpp && ./demo

# spiral classifier (macOS/clang with Homebrew libomp shown; drop the OpenMP flags if unneeded)
clang++ -std=c++23 -O3 -march=native spiral/spiral.cpp -o spiral_demo && ./spiral_demo

# MNIST training (see MNIST/MNIST.cpp header for the OpenMP variant)
g++ -std=c++23 -O3 -o mnist_train MNIST/MNIST.cpp && ./mnist_train

# tic-tac-toe agent benchmark
g++ -std=c++23 -O3 -o ttt_interface tictactoe/interface.cpp && ./ttt_interface
```

---

## Quick Start API Example

```cpp
#include "FFNN.hpp"
#include "AdamOptimiser.hpp"

int main() {
    std::vector<size_t> shape = {128, 256, 128};
    std::vector<ActivationFunc> acts = {ActivationFunc::relu, ActivationFunc::relu};

    FFNN model = FFNN::from_random_he_scaling(shape, acts);
    AdamOptimiser opt = AdamOptimiser::from_ffnn(model, CostType::mse, 1e-3f);

    Eigen::MatrixXf x = Eigen::MatrixXf::Random(128, 64);
    Eigen::MatrixXf y = Eigen::MatrixXf::Random(128, 64);

    for (int i = 0; i < 1000; ++i) {
        float cost = opt.step(x, y);
        if (i % 100 == 0) {
            std::cout << "step=" << i << " cost=" << cost << "\n";
        }
    }
}
```

With a `DataSet` and the shared trainer:

```cpp
#include "FFNN.hpp"
#include "AdamOptimiser.hpp"
#include "Trainer.hpp"

DataSet train_data = DataSet::from_file("train.dat");
DataSet test_data = DataSet::from_file("test.dat");

FFNN model = FFNN::from_random_he_scaling({784, 128, 10}, {ActivationFunc::relu, ActivationFunc::softmax});
AdamOptimiser opt = AdamOptimiser::from_ffnn(model, CostType::categorical_cross_entropy, 1e-3f);

TrainSettings settings{.num_steps = 10000, .batch_size = 256, .checkpoint_path = "best.dat"};
TrainResult result = train(model, opt, train_data, settings,
    [&](const FFNN& m) { return nn_utils::argmax_accuracy(m, test_data); });
```

Notes:

- Columns are batch elements throughout: inputs/targets/`DataSet` matrices are `dim x n`.
- Keep your input/target matrix shapes consistent with layer sizes.
- `forward`/`backward` are `const` and take a `ForwardWorkspace`/`BackpropWorkspace`; there's also
  a convenience `forward(input)` overload that allocates its own workspace for one-off calls.

---

## Included Example Projects

### `main.cpp` optimizer comparison

Runs two identical network setups in parallel — one with gradient descent, one with Adam — both
trained on random synthetic data. Useful for quickly sanity-checking update behavior; Adam should
converge faster than plain gradient descent on the same data.

### `spiral/spiral.cpp` spiral classifier

A self-contained demo: generates an n-arm spiral dataset in memory (no files needed), trains a
small classifier with `Trainer` and an exponential LR scheduler, and renders the learned decision
boundary as a coloured ASCII grid. A good first thing to run to confirm the toolchain works.

### `MNIST/MNIST.cpp` training on the MNIST/Fashion-MNIST binary datasets

What it does:

- loads `DataSet`s from `MNIST/bin/{train,test}.dat`,
- constructs a multi-layer FFNN,
- trains with Adam via the shared `Trainer`, with a per-batch noise-injection hook (see
  `MNIST/optimising.md` for why),
- checkpoints to `MNIST/models/best.dat` whenever test accuracy improves.

### `MNIST/Transform_Data_To_Binary.cpp` CSV to binary conversion

Reads `MNIST/MNIST/{train,test}.csv` and `MNIST/Fashion-MNIST/{train,test}.csv`, and writes the
binary `DataSet` files that `MNIST.cpp` reads (`MNIST/bin/*.dat` and
`MNIST/Fashion-MNIST/bin/*.dat`). Run this once before training if the `.dat` files don't exist
yet — they're gitignored.

### `tictactoe/` search and training experiments

This folder contains a board engine, search agents, and training experiments.

Current code centers on an `N x N` board with a configurable win length and includes:

- a reusable `TicTacToe<N, W>` game type,
- random, human, and minimax agents (`MinimaxRev1` through `MinimaxRev4`),
- benchmarking utilities for agent-vs-agent matches (`agentTools.hpp`),
- an `FFNNAgent` that uses the neural net as a board evaluator inside minimax,
- and training-data generation from self-play or minimax-vs-minimax games (`get_training_data`).

`tictactoe/interface.cpp` benchmarks search agents. `tictactoe/train.cpp` is currently an
incomplete stub.

---

## Model and Dataset Files

`FFNN` supports versioned binary persistence:

- save: `model.write_to_file("model.dat")`
- load: `FFNN loaded = FFNN::from_file("model.dat")`

The file starts with a `"FFNN"` magic and version number; files from before this format (no
magic header) are rejected with a clear error rather than silently misread — retrain or
regenerate them.

`DataSet` uses the same idea (`"FFDS"` magic + version):

- save: `dataset.write_to_file("data.dat")`
- load: `DataSet loaded = DataSet::from_file("data.dat")`
- `DataSet::from_files({...})` concatenates multiple files
- `DataSet::from_samples(inputs, labels)` builds one from a `vector<Eigen::VectorXf>` pair, for
  producers (like self-play) that don't know the sample count ahead of time

Both formats are gitignored (`*.dat`); regenerate them locally with
`MNIST/Transform_Data_To_Binary.cpp` or by rerunning training.

---

## Running the Tests

```bash
./tests/run_tests.sh
```

or, via CMake, `ctest --test-dir build`. Covers:

- `test_gradients.cpp`: `FFNN::backward` vs. central-difference numerical gradients across
  activation/cost/regularization combinations,
- `test_serialization.cpp`: `FFNN`/`DataSet` binary round trips + bad-magic rejection,
- `test_batcher.cpp`: `Batcher` seed determinism and batch-size clamping.

---

## Common Gotchas

- Activation/cost pairing matters:
	- `binary_cross_entropy` expects sigmoid in the output layer.
	- `categorical_cross_entropy` expects softmax in the output layer.
- Regularization is set on the optimizer, not on `FFNN::from_random_he_scaling`.
- File paths in examples are hardcoded relative paths; run executables from the repo root.
- Batch dimensions are column-oriented throughout (columns are samples).
- If OpenMP flags are unsupported on your compiler, compile without them first.

---

## Performance Tips

- Build with `-O3` (CMake's `Release` config does this) for meaningful training speed.
- Consider `-march=native` where available (CMake enables it automatically if supported).
- Reuse a `ForwardWorkspace`/`BackpropWorkspace` across calls instead of using the
  allocating convenience `forward(input)` overload in any hot loop.
- On some CPUs, denormals can hurt performance; consider enabling flush-to-zero in training code.
- If profiling shows I/O bottlenecks, cache parsed datasets in the binary `DataSet` format.
