Feedforward and recurrent machine learning in NumPy.
-----------------------------------------------------

In this repository, important machine learning structures are implemented 
in pure NumPy for educational purposes. Working only in NumPy makes the backward
pass explicit, which is automatically computed in frameworks like PyTorch.
Implementing the backward pass by hand is a very instructive exercise.

The ```feedforward``` folder explores fully connected and convolutional
layers on the MNIST digit recognition problem.

The ```rnn``` folder contains recurrent neural networks and code to train them
on the adding problem.

The source code in this repository runs using NumPy versions >= 1.18.1
and should run reasonably well on any mid to top range CPU.

I am looking for feedback! If you see an error please open an issue.
