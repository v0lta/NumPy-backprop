Numpy Re-Implementations of iconic machine learning breakthroughs.
------------------------------------------------------------------

In this repository, important machine learning structures are implemented 
in pure NumPy for educational purposes. Working only in NumPy makes the backward
pass explicit, which is automatically computed in frameworks like pyTorch.
Implementing the backward pass by hand was a very instructive exercise.

The 'feedforward' folder explores fully connected and convolutional
layers on the Mnist problem.

The 'rnn' folder contains recurrent neural networks and code to train them
on the adding problem.

The source code in this repository has been tested using NumPy version 1.18.1
and should run on any mid to top range CPU.


Sample output Rummelhard Fig. 1:
-------------------------
```
[1. 1. 1. 0. 0. 0.] [0.] 0.0 0.0
[0. 1. 0. 1. 1. 1.] [0.] 0.0 0.0
[0. 0. 0. 1. 1. 0.] [0.] 0.0 0.0
[0. 1. 0. 0. 1. 1.] [0.] 0.0 0.0
[0. 0. 1. 1. 1. 0.] [0.] 0.0 0.0
[1. 1. 1. 1. 1. 0.] [0.] 0.0 0.0
[1. 0. 0. 0. 1. 1.] [0.] 0.0 0.0
[0. 1. 0. 0. 0. 0.] [0.] 0.0 0.0
[0. 1. 1. 0. 0. 1.] [0.] 0.0 0.0
[1. 1. 1. 1. 0. 0.] [0.] 0.0 0.0
[1. 1. 0. 1. 0. 1.] [0.] 0.0 0.0
[1. 0. 1. 1. 1. 1.] [0.] 0.0 0.0
[0. 0. 1. 0. 0. 1.] [0.] 0.0 0.0
[1. 0. 0. 0. 0. 1.] [1.] 1.0 0.0
[1. 0. 0. 0. 0. 1.] [1.] 1.0 0.0
[1. 0. 1. 0. 1. 0.] [0.] 0.0 0.0
[1. 1. 1. 1. 1. 1.] [1.] 1.0 0.0
[0. 0. 0. 1. 0. 0.] [0.] 0.0 0.0
[1. 0. 1. 0. 1. 0.] [0.] 0.0 0.0
[0. 1. 0. 0. 0. 1.] [0.] 0.0 0.0
[ -7.04850922 -14.02958679   3.52852193  -3.53537362  14.02009255
   7.03436972  -2.35086203]
[  7.00471273  13.95149336  -3.50750519   3.50812809 -13.98073521
  -7.006003    -2.32103631]
[-13.3012845  -13.32363225   6.1053583 ]
```
