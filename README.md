# 1 step

Cloned the rusty-machine repo https://github.com/AtheMathmo/rusty-machine

I chose this repo from this collection https://www.arewelearningyet.com/neural-networks/ because i looked for a minimal framework that implements the utility of feedforward neural network training without deep learning approaches. This repo seemed to be the most tidied. Unfortunately the Levenberg marquardt optimizer is not implemented, but maybe i can add it. As i did not want to use wrappers, this is the best implementation for my usecase.

The first test showed that the framework is able to train on an AND gate with a BCECriterion and sigmoid activation functio. Here are some pics:

<img src="img/2-5-5-2-1.png" alt="2-5-5-2-1" width="500">

And with an unsufficient network structure:

<img src="img/2-1-1-1.png" alt="2-5-5-2-1" width="500">

You can see the threshold of the AND gate at 0.7

This was all done in the example/test.rs which was adjusted from the nnet-and_gate.rs

# Step 2

Now i want to use a 1D benchmark function in the [-1,1] range to test the capability to approximate functions with a MSE (mean square error) error metric. Is is commonly used for fitting. For the AND gate a BCE (binary cross entropy) was used that is an error function for classification.

## Benchmark function implementation

