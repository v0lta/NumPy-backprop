# based on https://gist.github.com/karpathy/d4dee566867f8291f086,
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/neuralnet.py
# see also https://arxiv.org/pdf/1503.04069.pdf

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
        # block input
        zbar = np.matmul(self.Wz, x) + np.matmul(self.Rz, h) + self.bz
        z = self.state_activation(zbar)
        # input gate
        ibar = np.matmul(self.Wi, x) + np.matmul(self.Ri, h) + self.pi*c \
            + self.bi
        i = self.gate_i_act(ibar)
        # forget gate
        fbar = np.matmul(self.Wf, x) + np.matmul(self.Rf, h) + self.pf*c \
            + self.bf
        f = self.gate_f_act(fbar)
        # cell
        c = z * i + c * f
        # output gate
        obar = np.matmul(self.Wo, x) + np.matmul(self.Ro, h) + self.po*c \
            + self.bo
        o = self.gate_o_act(obar)
        # block output
        y = self.out_activation(c)*o
        # linear projection
        y = np.matmul(self.Wout, y) + self.bout
        return c, h, y, zbar, ibar, fbar, obar

    def backward(self, x, h, zbar, ibar, fbar, obar, c, cm1,
                 deltay, deltac, deltao, deltai, deltaf):
        # projection backward
        dWout = np.matmul(deltay, np.transpose(h, [0, 2, 1]))
        dbout = 1*deltay
        deltay = np.matmul(np.transpose(self.Wout, [0, 2, 1]), deltay)

        # block backward
        deltao = deltay * self.out_activation(c)\
            * self.gate_o_act.backward(obar)
        deltac = deltay * self.gate_o_act(obar)\
            * self.state_activation.backward(c)\
            + self.po*deltao \
            + self.pi*deltai \
            + self.pf*deltaf \
            + deltac*self.gate_f_act(fbar)
        deltaf = deltac * cm1 * self.gate_f_act.backward(fbar)
        deltai = deltac * self.state_activation(zbar)
        deltaz = deltac*self.gate_i_act(ibar)\
            * self.state_activation.backward(zbar)

        # weight backward
        dWz = np.matmul(deltaz, np.transpose(x, [0, 2, 1]))
        dWi = np.matmul(deltai, np.transpose(x, [0, 2, 1]))
        dWf = np.matmul(deltaf, np.transpose(x, [0, 2, 1]))
        dWo = np.matmul(deltao, np.transpose(x, [0, 2, 1]))

        dRz = np.matmul(deltaz, np.transpose(h, [0, 2, 1]))
        dRi = np.matmul(deltai, np.transpose(h, [0, 2, 1]))
        dRf = np.matmul(deltaf, np.transpose(h, [0, 2, 1]))
        dRo = np.matmul(deltao, np.transpose(h, [0, 2, 1]))

        dbz = deltaz
        dbi = deltai
        dbf = deltaf
        dbo = deltao

        dpi = c*deltai
        dpf = c*deltaf
        dpo = c*deltao

        return deltay, dWout, dbout, deltao, deltaf, deltai, deltaz,\
            dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dbz, dbi, dbf, dbo,\
            dpi, dpf, dpo
