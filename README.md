# Deep Learning Basics Numpy Implementation

This repo includes my own implementations of some deep learning basics: **activation functions**, **SoftmaxCrossEntrop** loss function using LogSumExp trick, **Bath Normalization**, etc. These implementations used python Numpy library only and include both forward pass and backprop algorithm. 

Also, I implemented a **Multi-Layer Perceptron** with an API similar to popular Automatic Differentiation Libraries like PyTorch. This generic implementation supports an arbitrary number of layers, types of activations, network sizes, batch norm layer and momentum.


### Note:
The parameters for my MLP implementaion are:

• input_size: The size of each individual data example.

• output_size: The number of outputs.

• hiddens: A list with the number of units in each hidden layer.

• activations: A list of Activation objects for each layer.

• weight_init_fn: A function applied to each weight matrix before training.

• bias_init_fn: A function applied to each bias vector before training.

• criterion: A Criterion object to compute the loss and its derivative.

• lr: The learning rate.

• momentum: Momentum scale.

• num_bn_layers: Number of BatchNorm layers start from upstream.


