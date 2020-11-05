Feedforward and recurrent machine learning in NumPy.
-----------------------------------------------------

In this repository, important machine learning structures are implemented 
in pure NumPy for educational purposes. Working only in NumPy makes the backward
pass explicit, which is automatically computed in frameworks like PyTorch.
Implementing the backward pass by hand is a very instructive exercise.
Pure python is not very efficient on it's own, this code is purely educational
and not suitable for production purposes.

The ```feedforward``` folder explores fully connected and convolutional
layers on the MNIST digit recognition problem.

The ```rnn``` folder contains recurrent neural networks and code to train them
on adding and memory problems similar to those described by Hochreiter und
Schmidhuber in 1997.

The source code in this repository runs using NumPy versions >= 1.18.1
and should run reasonably well on any mid to top range CPU.

I am looking for feedback! If you see an error please open an issue.
