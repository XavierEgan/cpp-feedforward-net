

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
