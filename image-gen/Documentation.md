# Image Generation
## Goal
Get a ffnn with my library to produce an image in the same size and style of MNIST from an input vector of 10 inputs, with one being 1 and the rest being 0.
Example:
input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
output: image of 2 in 28 x 28 space.

## Background
This document will document my process for producing an ai model to perform this task.

### Terminology
"an input of 3" = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


## Idea 1:
Two models, MNIST predictor and Image generator (M and G).
G generates an image with an input of n, M predicts the number.
Backprop gets run backwards through M to G on the difference between the predicted number and the input.

### Expected problems
There is nothing encouraging G to generate coherent images, it will likely learn seemingly random jibberish that happens to strongly activate the predictor towards the answer.

### Results
