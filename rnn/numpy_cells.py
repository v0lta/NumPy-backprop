# based on https://gist.github.com/karpathy/d4dee566867f8291f086
# and https://github.com/wiseodd/hipsternet/blob/master/hipsternet/neuralnet.py

import numpy as np


class MSELoss(object):
    ''' Mean squared error loss function. '''
    def forward(self, label, out):
        diff = out - label
        return np.mean(0.5*diff*diff)

    def backward(self, label, out):
        return out - label


class Tanh(object):
    def forward(self, inputs):
        return np.tanh(inputs)

    def backward(self, inputs, delta):
        return (1. - np.tanh(inputs)*np.tanh(inputs))*delta


class BasicCell(object):
    """Basic (Elman) rnn cell."""

    def __init__(self, hidden_size=250, input_size=1, output_size=1,
                 activation=Tanh()):
        self.hidden_size = hidden_size
        # input to hidden
        self.Wxh = np.random.randn(1, hidden_size, input_size)*0.01
        # hidden to hidden
        self.Whh = np.random.randn(1, hidden_size, hidden_size)*0.01
        # hidden to output
        self.Why = np.random.randn(1, output_size, hidden_size)*0.01
        # hidden bias
        self.bh = np.zeros((1, hidden_size, 1))
        # output bias
        self.by = np.zeros((1, output_size, 1))
        self.activation = activation

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.hidden_size, 1))

    def forward(self, x, h):
        h = np.matmul(self.Whh, h) + np.matmul(self.Wxh, x) + self.bh
        h = self.activation.forward(h)
        y = np.matmul(self.Why, h) + self.by
        return y, h

    def backward(self, deltay, deltah, x, h, y):
        # output backprop
        dydh = np.matmul(np.transpose(self.Why, [0, 2, 1]), deltay)
        dWhy = np.matmul(deltay, np.transpose(h, [0, 2, 1]))
        dby = 1*deltay

        delta = self.activation.backward(inputs=h, delta=dydh) + deltah
        # recurrent backprop
        dhdh = np.matmul(np.transpose(self.Whh, [0, 2, 1]), delta)
        dWxh = np.matmul(delta, np.transpose(x, [0, 2, 1]))
        dWhh = np.matmul(delta, np.transpose(h, [0, 2, 1]))
        dbh = 1*delta
        # deltah, dWhh, dWxh, dbh, dWhy, dby
        return dhdh, dWhh, dWxh, dbh, dWhy, dby


class LSTMcell(object):
    def __init__(self):
        pass

    def forward(self):
        pass