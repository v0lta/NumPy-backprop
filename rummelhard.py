# A reimplementation of the mirror symmetry example by rumelhart et al.
import numpy as np
import matplotlib.pyplot as plt
import ipdb
debug_here = ipdb.set_trace


def rumelhard_problem_rnd(length):
    assert length % 2 == 0
    # generate a binary vector
    problem_vector = np.random.randn(int(length))
    problem_vector = np.greater(problem_vector, 0).astype(np.float32)
    # check symmetry
    first_half, second_half = np.split(problem_vector, 2)
    symmetric = np.array_equal(first_half, np.flip(second_half, 0))
    return problem_vector, float(symmetric)


# todo: use and accumulate gradients over.
def rumelhard_problem(length):
    assert length % 2 == 0
    max_val = np.power(2, length)
    batch_lst = []
    for i in range(max_val):
        binary_str = np.binary_repr(i, length)
        binary_array = np.array([int(bstr) for bstr in binary_str])
        first_half, second_half = np.split(binary_array, 2)
        symmetric = np.array_equal(first_half, np.flip(second_half, 0))
        batch_lst.append((binary_array, float(symmetric)))
    return batch_lst


def get_batch(length, size):
    x_lst = []
    y_lst = []
    for e in range(size):
        x, y = rumelhard_problem_rnd(length)
        x_lst.append(x)
        y_lst.append(y)
    return np.array(x_lst), np.array(y_lst)


class SymNet(object):
    '''
    Numpy-Implementation of the three neuron symmetry
    detection network proposed in the Rumelhart
    paper https://www.nature.com/articles/323533a0 .
    '''

    def __init__(self, length):
        self.weight_left = np.random.uniform(-0.3, 0.3, length + 1)
        self.weight_right = np.random.uniform(-0.3, 0.3, length + 1)
        self.weight_top = np.random.uniform(-0.3, 0.3, 3)

        self.vel_grad_left = np.zeros(length + 1)
        self.vel_grad_right = np.zeros(length + 1)
        self.vel_grad_top = np.zeros(3)

    def forward(self, inputs):
        inputs = np.concatenate([inputs, [1]], axis=0)
        x_left = np.sum(self.weight_left * inputs)
        y_left = self.sigmoid(x_left)
        x_right = np.sum(self.weight_right * inputs)
        y_right = self.sigmoid(x_right)
        y_top = np.array([y_left, y_right, 1.])
        x_top = np.sum(self.weight_top * y_top)
        y_top = self.sigmoid(x_top)
        return [y_top, y_left, y_right]

    def backward(self, y_top, y_left, y_right, inputs, target):
        inputs = np.concatenate([inputs, [1]], axis=0)
        dE_dy_top = y_top - target

        # chain rule
        dE_dx_top = dE_dy_top * y_top * (1 - y_top)
        y_top = np.array([y_left, y_right, 1.])
        dE_dw_top = dE_dx_top * y_top

        dE_dy_left = np.sum(dE_dx_top * self.weight_top)
        dE_dx_left = dE_dy_left * y_left * (1 - y_left)
        dE_dw_left = dE_dx_left * inputs

        dE_dy_right = np.sum(dE_dx_top * self.weight_top)
        dE_dx_right = dE_dy_right * y_right * (1 - y_right)
        dE_dw_right = dE_dx_right * inputs

        return dE_dw_top, dE_dw_left, dE_dw_right

    def sigmoid(self, x):
        return (1.0/(1.0 + np.exp(-x)))

    def loss(self, x, target):
        return 0.5*(np.sum((x - target)**2))

    def update(self, top_grad, left_grad,
               right_grad, step_size, vel_weight):
        self.vel_grad_left = -step_size*left_grad + vel_weight*self.vel_grad_left
        self.vel_grad_right = -step_size*right_grad + vel_weight*self.vel_grad_right
        self.vel_grad_top = -step_size*top_grad + vel_weight*self.vel_grad_top

        self.weight_left += self.vel_grad_left
        self.weight_right += self.vel_grad_right
        self.weight_top += self.vel_grad_top


# network parameters
length = 6

net = SymNet(length)

# gradient steps sizes
epsilon = 0.1
alpha = (1.0 - epsilon)

iterations = 64*1425
loss_lst = []
for i in range(iterations):
    inputs, targets = get_batch(length, 1)
    inputs = np.squeeze(inputs)
    top, left, right = net.forward(inputs)
    loss = net.loss(top, targets)
    top_grad, left_grad, right_grad = net.backward(top, left, right, inputs,
                                                   targets)
    net.update(top_grad, left_grad, right_grad, epsilon, alpha)
    loss_lst.append(loss)

plt.plot(loss_lst)
plt.show()

# test the results
for i in range(20):
    inputs, targets = get_batch(length, 1)
    inputs = np.squeeze(inputs)
    y_top, y_left, y_right = net.forward(inputs)
    loss = net.loss(y_top, targets)
    print(inputs, targets, np.round(y_top, 1), np.round(loss, 3))

print(net.weight_left)
print(net.weight_right)
print(net.weight_top)
