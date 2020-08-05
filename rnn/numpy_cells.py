# based on https://gist.github.com/karpathy/d4dee566867f8291f086
# and https://github.com/wiseodd/hipsternet/blob/master/hipsternet/neuralnet.py

import numpy as np


class BasicCell(object):
    """Basic (Elman) rnn cell."""

    def __init__(self, hidden_size=250, input_size=1, activation=None):
        # model parameters
        # input to hidden
        self.Wxh = np.random.randn(hidden_size, input_size)*0.1
        # hidden to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size)*0.1
        # hidden to output
        self.Why = np.random.randn(input_size, hidden_size)*0.1
        # hidden bias
        self.bh = np.zeros((hidden_size, 1))
        # output bias
        self.by = np.zeros((input_size, 1))

    def forward(self, x, h):
        h = None
        return None
