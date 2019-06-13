'''
A basic convolution test case.
'''

import numpy as np


class basic_multiplication(object):
    def __init__(self):
        self.size = 50
        self.step = 1
        self.weights = np.random.randn(self.size)

    def __call__(self, inputs):
        assert len(inputs.shape) == 1
        return inputs*self.weights


# linear multiplication example
inputs = np.ones(50)
mul = basic_multiplication()
print(mul.weights)

target = np.sin(np.linspace(0, 2*np.pi))


def cost_function(res, targets):
    return 0.5*np.mean((res - targets)**2)


def cost_dev(res, targets, inputs):
    return (res - targets)*inputs


iterations = 100
for i in range(iterations):

    # forward pass
    res = mul(inputs)
    cost = cost_function(res, target)

    # backward pass
    grad = cost_dev(res, target, inputs)
    mul.weights = mul.weights - 0.1*grad
    print(np.linalg.norm(mul.weights), cost)
print(mul.weights)
print(res)
