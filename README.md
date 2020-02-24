# Deep Learning Basics Numpy Implementation

## MLP.py
This file includes my own implementations of some deep learning basics: **activation functions**, **SoftmaxCrossEntrop** loss function using LogSumExp trick, **Bath Normalization**, etc. These implementations used python Numpy library only and include both forward pass and backprop algorithm. 

Then, I implemented a **Multi-Layer Perceptron** with an API similar to popular Automatic Differentiation Libraries like PyTorch. This generic implementation supports an arbitrary number of layers, types of activations, network sizes, batch norm layer and momentum.

## CNN.py
A generic implementation of **CNN** which has similar usage and functionality to torch.nn.Conv1d.

## GRU.py
A numpy implementation of **GRU** forward and backprop, similar to the Pytorch equivalent nn.GRUCell.

## BeamSearch.py
Implementation of **greedy search** and **beam search**

Greedy search greedily picks the label with maximum probability at each time step to compose the output sequence. Beam search is a more effective decoding technique to to obtain a sub-optimal result out of sequential decisions, striking a balance between a greedy search and an exponential exhaustive search by keeping a beam of top-k scored sub-sequences at each time step.

NOTE:
For both the functions, the *SymbolSets* is a list of symbols that can be predicted except for the blank symbol; *y_probs*, an array of shape (len(SymbolSets) + 1 , seq length, batch size ), is the probability distribution over all symbols including the blank symbol at each time step (note that probability of blank for all time steps is the first row of y probs ).
