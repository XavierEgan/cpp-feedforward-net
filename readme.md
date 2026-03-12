# Cpp Feedforward Neural Net

A lightweight C++ feedforward neural network project built around `Eigen::MatrixXf`.

This repository is focused on:

- learning and experimenting with core neural-network training loops,
- keeping the code easy to inspect and tweak,
- and providing a few concrete example projects (toy benchmark, MNIST/Fashion-MNIST, tic-tac-toe scaffold).

It is intentionally simple and header-driven, so you can read the implementation directly and modify it quickly.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Current Features](#current-features)
- [Repository Layout](#repository-layout)
- [Dependencies](#dependencies)
- [Build and Run](#build-and-run)
- [Quick Start API Example](#quick-start-api-example)
- [Included Example Projects](#included-example-projects)
- [Model Saving and Loading](#model-saving-and-loading)
- [Design Notes](#design-notes)
- [Common Gotchas](#common-gotchas)
- [Performance Tips](#performance-tips)
- [Where to Extend Next](#where-to-extend-next)

---

## What This Project Does

At a high level, this repo gives you a small neural-net training framework in modern C++.

You define:

- network shape (e.g. `{784, 256, 128, 10}`),
- activation functions per non-input layer,
- cost/loss function,
- optimizer,
- and training loop behavior.

The core `FFNN` type handles forward propagation, backpropagation, and parameter storage.

Optimizers then apply updates to weights/biases:

- plain gradient descent,
- Adam.

There are also small utilities for:

- random mini-batch generation,
- activation + loss math,
- and learning-rate scheduling.

---

## Current Features

### Network

- Dense feedforward architecture (`FFNN`).
- He-style random initialization (`from_random_he_scaling`).
- Binary serialization/deserialization (`write_to_file`, `from_file`).

### Activations

- `linear`
- `relu`
- `relu_clipped`
- `sigmoid`
- `softmax`

### Costs

- `quadratic`
- `binary_cross_entropy`
- `categorical_cross_entropy`

### Regularization

- `none`
- `l1`
- `l2`

### Optimizers

- `GradientDescentOptimiser`
- `AdamOptimiser`

### LR Schedulers

- linear decay
- exponential decay
- decay-on-plateau

---

## Repository Layout

Top-level highlights:

- `FFNN.hpp`: neural-net definition + forward/backward logic
- `ActivationFunction.hpp`: activations + derivatives
- `CostType.hpp`: loss functions + derivatives
- `GradientDescentOptimiser.hpp`: SGD-like update step
- `AdamOptimiser.hpp`: Adam update step
- `LrSchedulers.hpp`: learning-rate scheduler helpers
- `NN_Utils.hpp`: batching + shape checks + utility math
- `main.cpp`: simple optimizer comparison demo
- `MNIST/MNIST.cpp`: training example on CSV image data
- `MNIST/MNIST_inference.cpp`: inference/evaluation example
- `tictactoe/`: early scaffold for game-based training ideas
- `Eigen/`: vendored Eigen headers

---

## Dependencies

This project currently depends on:

- a C++23-capable compiler,
- Eigen headers (already included in this repo under `Eigen/`),
- optional OpenMP support for some training runs.

No external package manager is required for the core code path shown here.

---

## Build and Run

Since this is a header-only project, compilation of examples is done with direct command-line calls to `g++` or `clang++`. 

### 1) Basic Demo (`main.cpp`)

#### Windows (MinGW g++)

```bash
g++ -std=c++23 -O3 -o demo.exe main.cpp
demo.exe
```

#### Linux/macOS (g++)

```bash
g++ -std=c++23 -O3 -o demo main.cpp
./demo
```

### 2) MNIST / Fashion-MNIST Training Example

#### Windows (without OpenMP)

```bash
g++ -std=c++23 -O3 -o mnist_train.exe MNIST/MNIST.cpp
mnist_train.exe
```

#### Windows (with OpenMP)

```bash
g++ -std=c++23 -fopenmp -O3 -o mnist_train.exe MNIST/MNIST.cpp
mnist_train.exe
```

#### macOS (clang + Homebrew libomp)

```bash
clang++ -std=c++23 -O3 -march=native -Xpreprocessor -fopenmp \
	-I"$(brew --prefix libomp)/include" MNIST/MNIST.cpp \
	-L"$(brew --prefix libomp)/lib" -lomp -o mnist_train
./mnist_train
```

### 3) MNIST Inference Example

```bash
g++ -std=c++23 -O3 -o mnist_infer.exe MNIST/MNIST_inference.cpp
mnist_infer.exe
```

### 4) Tic-Tac-Toe Training Scaffold

```bash
g++ -std=c++23 -O3 -o ttt_train.exe tictactoe/train.cpp
ttt_train.exe
```

---

## Quick Start API Example

Use this if you want a minimal in-code setup:

```cpp
#include "FFNN.hpp"
#include "AdamOptimiser.hpp"

int main() {
		std::vector<size_t> shape = {128, 256, 128};
		std::vector<ActivationFunc> acts = {relu, relu};

		FFNN model = FFNN::from_random_he_scaling(shape, acts);
		AdamOptimiser opt(model, CostType::quadratic, 1e-3);

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

Notes:

- Columns are treated as batch elements in several training paths.
- Keep your input/target matrix shapes consistent with layer sizes.

---

## Included Example Projects

### `main.cpp` optimizer comparison

Runs two identical network setups in parallel:

- one with gradient descent,
- one with Adam,
- both trained on random synthetic data.

Useful for quickly sanity-checking update behavior.
You should see Adam converge faster (hard to see because it converges very fast), and GD achieving 

### `MNIST/MNIST.cpp` training on CSV datasets

What it does:

- reads CSV image data into vectors of `(image, one-hot label)`,
- constructs a multi-layer FFNN,
- samples random mini-batches,
- trains with Adam and a scheduler,
- periodically evaluates + writes model checkpoints (`.dat`).

Default data paths in code currently point to:

- `MNIST/Fashion-MNIST/train.csv`
- `MNIST/Fashion-MNIST/test.csv`

### `MNIST/MNIST_inference.cpp` model evaluation

What it does:

- loads a serialized model with `FFNN::from_file(...)`,
- reads test CSV data,
- runs forward passes,
- reports classification accuracy.

If your filenames differ, update the hardcoded paths in the source file.

### `tictactoe/` scaffold

This folder appears to be an early framework for self-play style training experimentation.

Current `train.cpp` is a minimal placeholder and can be used as a starting point for:

- board encoding,
- policy/value targets,
- and game-loop data generation.

---

## Model Saving and Loading

`FFNN` supports binary persistence:

- save: `model.write_to_file("model.dat")`
- load: `FFNN loaded = FFNN::from_file("model.dat")`

Serialized content includes:

- network shape,
- activation-function sequence,
- all weight matrices,
- all bias vectors.

This makes it straightforward to separate training and inference programs.

---

## Design Notes

- Uses `Eigen::MatrixXf` throughout (single precision).
- Focuses on readability and experimentation over framework-level abstraction.
- Most functionality is in headers, so stepping through in a debugger is simple.
- Validation checks exist for shape/layer assumptions and some cost/activation pairings.

---

## Common Gotchas

- Activation/cost pairing matters:
	- `binary_cross_entropy` expects sigmoid in output layer.
	- `categorical_cross_entropy` expects softmax in output layer.
- File paths in examples are hardcoded; adjust if your dataset names differ.
- Batch dimensions are column-oriented in this implementation style.
- If OpenMP flags are unsupported on your compiler, compile without them first.

---

## Performance Tips

- Build with `-O3` for meaningful training speed.
- Consider `-march=native` where available.
- On some CPUs, denormals can hurt performance; this repo includes examples of enabling flush-to-zero in training code.
- If profiling shows I/O bottlenecks, cache parsed datasets in a binary format.

---

## Where to Extend Next

Good next improvements if you want to keep building this out:

- add a `CMakeLists.txt` for easier cross-platform builds,
- add train/val split and early stopping,
- add mini-batch iterator classes instead of one-shot random sampling,
- add additional layers (dropout, batch norm),
- add unit tests for forward/backward correctness,
- add command-line flags for dataset path and hyperparameters.

---

If you are exploring neural nets from first principles in C++, this codebase is a solid practical sandbox: compact enough to understand end-to-end, but capable enough to run real training loops.
