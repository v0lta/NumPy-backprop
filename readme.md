Numpy Re-Implementations of iconic machine learning breakthroughs.
------------------------------------------------------------------

In this repository, important machine learning structures are implemented 
in pure NumPy for educational purposes. Working only in NumPy makes the backward
pass explicit, which is automatically computed in frameworks like PyTorch.
Implementing the backward pass by hand was a very instructive exercise.

The ```feedforward``` folder explores fully connected and convolutional
layers on the MNIST digit recognition problem.

The ```rnn``` folder contains recurrent neural networks and code to train them
on the adding problem.

The ```rummelhard``` folder revisits the symmetry detection problem from
the original paper on backpropagation.

The source code in this repository runs using NumPy versions >= 1.18.1
and should run on any mid to top range CPU.


