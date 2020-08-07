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


class Sigmoid(object):

    def sigmoid(self, inputs):
        return np.exp(inputs)/(1 + np.exp(inputs))

    def forward(self, inputs):
        return self.sigmoid(inputs)

    def backward(self, inputs, delta):
        return self.sigmoid(inputs)*(1 - self.sigmoid(inputs))*delta


class BasicCell(object):
    """Basic (Elman) rnn cell."""

    def __init__(self, hidden_size=250, input_size=1, output_size=1,
                 activation=Tanh()):
        self.hidden_size = hidden_size
        # input to hidden
        self.Wxh = np.random.randn(1, hidden_size, input_size)
        self.Wxh = self.Wxh / np.sqrt(hidden_size)
        # hidden to hidden
        self.Whh = np.random.randn(1, hidden_size, hidden_size)
        self.Whh = self.Whh / np.sqrt(hidden_size)
        # hidden to output
        self.Why = np.random.randn(1, output_size, hidden_size)
        self.Why = self.Why / np.sqrt(hidden_size)
        # hidden bias
        self.bh = np.zeros((1, hidden_size, 1))
        # output bias
        self.by = np.random.randn(1, output_size, 1)*0.01
        self.activation = activation

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.hidden_size, 1))

    def forward(self, x, h):
        h = np.matmul(self.Whh, h) + np.matmul(self.Wxh, x) + self.bh
        h = self.activation.forward(h)
        y = np.matmul(self.Why, h) + self.by
        return y, h

    def backward(self, deltay, deltah, x, h, hm1, y):
        # output backprop
        dydh = np.matmul(np.transpose(self.Why, [0, 2, 1]), deltay)
        dWhy = np.matmul(deltay, np.transpose(h, [0, 2, 1]))
        dby = 1*deltay

        delta = self.activation.backward(inputs=h, delta=dydh) + deltah
        # recurrent backprop
        dWxh = np.matmul(delta, np.transpose(x, [0, 2, 1]))
        dWhh = np.matmul(delta, np.transpose(hm1, [0, 2, 1]))
        dbh = 1*delta
        deltah = np.matmul(np.transpose(self.Whh, [0, 2, 1]), delta)
        # deltah, dWhh, dWxh, dbh, dWhy, dby
        return deltah, dWhh, dWxh, dbh, dWhy, dby


class LSTMcell(object):
    def __init__(self, hidden_size=250,
                 input_size=1, output_size=1):
        self.hidden_size = hidden_size
        # create the weights
        s = np.sqrt(hidden_size)
        self.Wz = np.random.randn(1, hidden_size, input_size)*s
        self.Wi = np.random.randn(1, hidden_size, input_size)*s
        self.Wf = np.random.randn(1, hidden_size, input_size)*s
        self.Wo = np.random.randn(1, hidden_size, input_size)*s

        self.Rz = np.random.randn(1, hidden_size, hidden_size)*s
        self.Ri = np.random.randn(1, hidden_size, hidden_size)*s
        self.Rf = np.random.randn(1, hidden_size, hidden_size)*s
        self.Ro = np.random.randn(1, hidden_size, hidden_size)*s

        self.bz = np.random.randn(1, hidden_size, 1)*s
        self.bi = np.random.randn(1, hidden_size, 1)*s
        self.bf = np.random.randn(1, hidden_size, 1)*s
        self.bo = np.random.randn(1, hidden_size, 1)*s

        self.pi = np.random.randn(1, hidden_size, 1)*s
        self.pf = np.random.randn(1, hidden_size, 1)*s
        self.po = np.random.randn(1, hidden_size, 1)*s

        self.state_activation = Tanh()
        self.out_activation = Tanh()
        self.gate_i_act = Sigmoid()
        self.gate_f_act = Sigmoid()
        self.gate_o_act = Sigmoid()

        self.Wout = np.random.randn(1, hidden_size, output_size)*s
        self.bout = np.random.randn(1, output_size, 1)

    def forward(self, x, h, c, cm1):
        z = np.matmul(self.Wz, x) + np.matmul(self.Rz, h) + self.bz
        z = self.state_activation(z)
        i = np.matmul(self.Wi, x) + np.matmul(self.Ri, h) + self.pi*c + self.bi
        i = self.gate_i_act(i)
        f = np.matmul(self.Wf, x) + np.matmul(self.Rf, h) + self.pf*c + self.bf
        f = self.gate_f_act(f)
        o = np.matmul(self.Wo, x) + np.matmul(self.Ro, h) + self.po*c + self.bo
        o = self.gate_o_act(o)
        c = z * i + c * f
        h = self.out_activation(c)*o
        y = np.matmul(self.Wout, h) + self.bout
        return c, h, y

    def backward(self, x, h, c, deltay, deltac):
        dydh = np.matmul(np.transpose(self.Wout, [0, 2, 1]), deltay)
        dWout = np.matmul(deltay, np.transpose(h, [0, 2, 1]))
        dbout = 1*deltay

        
        return None
