

# Profiling
We see from this Flame graph that the runtime is dominated by += and get_random_batch()
![alt text](image.png).

First lets investigate the +=. It's likely the += 