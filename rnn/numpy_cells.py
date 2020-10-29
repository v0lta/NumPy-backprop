# Created by moritz (wolter@cs.uni-bonn.de)

# based on https://gist.github.com/karpathy/d4dee566867f8291f086,
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/neuralnet.py
# see also https://arxiv.org/pdf/1503.04069.pdf

import numpy as np
import sys
sys.path.append("./feedforward/")
from numpy_layer import DenseLayer, MSELoss, CrossEntropyCost
from numpy_layer import Sigmoid

class Tanh(object):
    """ Hyperbolic tangent activation function. """
    def forward(self, inputs):
        return np.tanh(inputs)

    def backward(self, inputs, delta):
        return (1. - np.tanh(inputs)*np.tanh(inputs))*delta

    def prime(self, inputs):
        return (1. - np.tanh(inputs)*np.tanh(inputs))


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
        """Basic Cell forward pass.

        Args:
            x (np.array): The input at the current time step.
            h (np.array): The cell-state at the current time step.

        Returns:
            y (np.array): Cell output.
            h (np.array): Updated cell state.
        """
        h = np.matmul(self.Whh, h) + np.matmul(self.Wxh, x) + self.bh
        h = self.activation.forward(h)
        y = np.matmul(self.Why, h) + self.by
        return y, h

    def backward(self, deltay, deltah, x, h, hm1):
        """The backward pass of the Basic-RNN cell.

        Args:
            deltay (np.array): Deltas from the layer above.
                               [batch_size, output_size, 1].
            deltah (np.array): Cell state deltas.
            x (np.array): Input at current time step.
            h (np.array): State at current time step.
            hm1 (np.array): State at previous time step.

        Returns:
            deltah (np.array): Updated block deltas.
            dWhh (np.array): Recurrent weight matrix gradients.
            dWxh (np.array): Input weight matrix gradients
            dbh (np.array): Bias gradients.
            dWhy (np.array):  Output projection matrix gradients.
            dby (np.array): Ouput bias gradients.
        """
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
    def __init__(self, hidden_size=64,
                 input_size=1, output_size=1):
        """ Instantiate a Long Short Term Memory Cell.

        Args:
            hidden_size (int, optional): The cell size. Defaults to 64.
            input_size (int, optional): Input size. Defaults to 1.
            output_size (int, optional): The size of the output. Defaults to 1.
        """
        self.hidden_size = hidden_size
        # create the weights
        s = 1./np.sqrt(hidden_size)
        self.weights = {}
        self.weights['Wz'] = np.random.randn(1, hidden_size, input_size)*s
        self.weights['Wi'] = np.random.randn(1, hidden_size, input_size)*s
        self.weights['Wf'] = np.random.randn(1, hidden_size, input_size)*s
        self.weights['Wo'] = np.random.randn(1, hidden_size, input_size)*s

        self.weights['Rz'] = np.random.randn(1, hidden_size, hidden_size)*s
        self.weights['Ri'] = np.random.randn(1, hidden_size, hidden_size)*s
        self.weights['Rf'] = np.random.randn(1, hidden_size, hidden_size)*s
        self.weights['Ro'] = np.random.randn(1, hidden_size, hidden_size)*s

        self.weights['bz'] = np.zeros((1, hidden_size, 1))*s
        self.weights['bi'] = np.random.randn(1, hidden_size, 1)*s
        self.weights['bf'] = np.random.randn(1, hidden_size, 1)*s
        self.weights['bo'] = np.random.randn(1, hidden_size, 1)*s

        self.weights['pi'] = np.random.randn(1, hidden_size, 1)*s
        self.weights['pf'] = np.random.randn(1, hidden_size, 1)*s
        self.weights['po'] = np.random.randn(1, hidden_size, 1)*s

        self.block_act = Tanh()
        self.out_activation = Tanh()
        self.gate_i_act = Sigmoid()
        self.gate_f_act = Sigmoid()
        self.gate_o_act = Sigmoid()

        self.weights['Wout'] = np.random.randn(1, output_size, hidden_size)*s
        self.weights['bout'] = np.random.randn(1, output_size, 1)

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.hidden_size, 1))

    def forward(self, x, h, c) -> {}:
        """ Implementation of the LSTM forward pass.

        Args:
            x (np.array): Array containing the current inputs..
            h (np.array): Pre-projection cell output vector.
            c (np.array): Cell state.

        Returns:
            A dictionary containing:
                y (np.array): Projected cell output.
                c (np.array): Cell memory state
                h (np.array): Gated output vector.
                zbar (np.array): Pre-activation block input.
                z    (np.array): Block input vector.
                ibar (np.array): Pre-activation input gate vector.
                i    (np.array): Input gate vector.
                fbar (np.array): Pre-activation forget gate vector.
                f    (np.array): Forget gate vector.
                obar (np.array): Pre-activation output gate vector.
                o    (np.array): Output gate vector.
                x    (np.array): Input used to evaluate the cell.
        """
        # block input
        zbar = np.matmul(self.weights['Wz'], x) \
            + np.matmul(self.weights['Rz'], h) \
            + self.weights['bz']
        z = self.block_act.forward(zbar)
        # input gate
        ibar = np.matmul(self.weights['Wi'], x) \
            + np.matmul(self.weights['Ri'], h) \
            + self.weights['pi']*c \
            + self.weights['bi']
        i = self.gate_i_act.forward(ibar)
        # forget gate
        fbar = np.matmul(self.weights['Wf'], x) \
            + np.matmul(self.weights['Rf'], h) \
            + self.weights['pf']*c \
            + self.weights['bf']
        f = self.gate_f_act.forward(fbar)
        # cell
        c = z * i + c * f
        # output gate
        obar = np.matmul(self.weights['Wo'], x) \
            + np.matmul(self.weights['Ro'], h) \
            + self.weights['po']*c \
            + self.weights['bo']
        o = self.gate_o_act.forward(obar)
        # block output
        h = self.out_activation.forward(c)*o
        # linear projection
        y = np.matmul(self.weights['Wout'], h) + self.weights['bout']
        return {'y': y, 'c': c, 'h': h, 'zbar': zbar, 'z': z,
                'ibar': ibar, 'i': i,  'fbar': fbar, 'f': f,
                'obar': obar, 'o': o, 'x': x}

    def backward(self, deltay, fd, prev_fd, prev_gd) -> {}:
        """ As described in https://arxiv.org/pdf/1503.04069.pdf section B.

        Args:
            deltay (np.array): Gradients from the layer above.
            fd (dict): Forward dictionary recording the forward pass values.
            prev_fd (dict): Dictionary at time t+1.
            prev_gd (dict): Previous gradients at time t+1.

        Returns:
            A dictionary with the gradients at time t.
        """
        # projection backward
        dWout = np.matmul(deltay, np.transpose(fd['h'], [0, 2, 1]))
        dbout = 1*deltay
        deltay = np.matmul(np.transpose(self.weights['Wout'], [0, 2, 1]),
                           deltay)

        # block backward
        deltah = deltay \
            + np.matmul(np.transpose(self.weights['Rz'], [0, 2, 1]),
                        prev_gd['deltaz']) \
            + np.matmul(np.transpose(self.weights['Ri'], [0, 2, 1]),
                        prev_gd['deltai']) \
            + np.matmul(np.transpose(self.weights['Rf'], [0, 2, 1]),
                        prev_gd['deltaf']) \
            + np.matmul(np.transpose(self.weights['Ro'], [0, 2, 1]),
                        prev_gd['deltao'])

        deltao = deltah * self.out_activation.forward(fd['c']) \
            * self.gate_o_act.prime(fd['obar'])
        deltac = deltah * fd['o'] * self.block_act.prime(fd['c'])\
            + self.weights['po']*deltao \
            + self.weights['pi']*prev_gd['deltai'] \
            + self.weights['pf']*prev_gd['deltaf'] \
            + prev_gd['deltac']*prev_fd['f']
        deltaf = deltac * prev_fd['c'] * self.gate_f_act.prime(fd['fbar'])
        deltai = deltac * fd['z'] * self.gate_i_act.prime(fd['ibar'])
        deltaz = deltac * fd['i'] * self.block_act.prime(fd['zbar'])

        # weight backward
        dWz = np.matmul(deltaz, np.transpose(fd['x'], [0, 2, 1]))
        dWi = np.matmul(deltai, np.transpose(fd['x'], [0, 2, 1]))
        dWf = np.matmul(deltaf, np.transpose(fd['x'], [0, 2, 1]))
        dWo = np.matmul(deltao, np.transpose(fd['x'], [0, 2, 1]))

        # Compute recurrent weight gradients.
        dRz = np.matmul(prev_gd['deltaz'], np.transpose(fd['h'], [0, 2, 1]))
        dRi = np.matmul(prev_gd['deltai'], np.transpose(fd['h'], [0, 2, 1]))
        dRf = np.matmul(prev_gd['deltaf'], np.transpose(fd['h'], [0, 2, 1]))
        dRo = np.matmul(prev_gd['deltao'], np.transpose(fd['h'], [0, 2, 1]))

        dbz = deltaz
        dbi = deltai
        dbf = deltaf
        dbo = deltao

        dpi = fd['c']*prev_gd['deltai']
        dpf = fd['c']*prev_gd['deltaf']
        dpo = fd['c']*deltao

        return {'deltac': deltac, 'deltaz': deltaz, 'deltao': deltao,
                'deltai': deltai, 'deltaf': deltaf,
                'dWout': dWout, 'dbout': dbout,
                'dWz': dWz, 'dWi': dWi, 'dWf': dWf, 'dWo': dWo,
                'dRz': dRz, 'dRi': dRi, 'dRf': dRf, 'dRo': dRo,
                'dbz': dbz, 'dbi': dbi, 'dbf': dbf, 'dbo': dbo,
                'dpi': dpi, 'dpf': dpf, 'dpo': dpo}

    def update(self):
        """ Compute a SGD update step. """
        pass


class GRU(object):
    def __init__(self, hidden_size=250,
                 input_size=1, output_size=1):
        """Create a Gated Recurrent unit.

        Args:
            hidden_size (int, optional): The cell size. Defaults to 250.
            input_size (int, optional): The number of input dimensions.
                                        Defaults to 1.
            output_size (int, optional): Output dimension number.
                                         Defaults to 1.
        """
        self.hidden_size = hidden_size
        # create the weights
        s = 1./np.sqrt(hidden_size)
        self.weights = {}
        self.weights['Vr'] = np.random.randn(1, hidden_size, input_size)*s
        self.weights['Vu'] = np.random.randn(1, hidden_size, input_size)*s
        self.weights['V'] = np.random.randn(1, hidden_size, input_size)*s

        self.weights['Wr'] = np.random.randn(1, hidden_size, hidden_size)*s
        self.weights['Wu'] = np.random.randn(1, hidden_size, hidden_size)*s
        self.weights['W'] = np.random.randn(1, hidden_size, hidden_size)*s

        self.weights['br'] = np.zeros((1, hidden_size, 1))*s
        self.weights['bu'] = np.random.randn(1, hidden_size, 1)*s
        self.weights['b'] = np.random.randn(1, hidden_size, 1)*s

        self.state_activation = Tanh()
        # self.out_activation = Tanh()
        self.gate_r_act = Sigmoid()
        self.gate_u_act = Sigmoid()

        self.weights['Wout'] = np.random.randn(1, output_size, hidden_size)*s
        self.weights['bout'] = np.random.randn(1, output_size, 1)

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.hidden_size, 1))

    def forward(self, x, h):
        """Gated recurrent unit forward pass.

        Args:
            x (np.array): Current input [batch_size, input_dim, 1]
            h (np.array): Current cell state [batch_size, hidden_dim, 1]

        Returns:
            A dictionary containing:
                y    (np.array): Current output
                h    (np.array): Current cell state
                zbar (np.array): Pre-activation state candidate values.
                z    (np.array): State candidate values
                hbar (np.array): Pre-activation block input.
                h    (np.array): Block input.
                rbar (np.array): Pre-activation reset-gate input.
                r    (np.array): Reset gate output vector
                ubar (np.array): Pre-activation update-gate input.
                u    (np.array): Update-gate output vector.
        """
        # reset gate
        rbar = np.matmul(self.weights['Vr'], x) \
            + np.matmul(self.weights['Wr'], h) \
            + self.weights['br']
        r = self.gate_r_act.forward(rbar)
        # update gate
        ubar = np.matmul(self.weights['Vu'], x) \
            + np.matmul(self.weights['Wu'], h) \
            + self.weights['b']
        u = self.gate_u_act.forward(ubar)
        # block input
        hbar = r*h
        zbar = np.matmul(self.weights['V'], x) \
            + np.matmul(self.weights['W'], hbar) \
            + self.weights['b']
        z = self.state_activation.forward(zbar)
        # recurrent update
        h = u*z + (1 - u)*h
        # linear projection
        y = np.matmul(self.weights['Wout'], h) + self.weights['bout']
        return {'y': y, 'x': x,
                'hbar': hbar, 'h': h,
                'zbar': zbar, 'z': z,
                'rbar': rbar, 'r': r,
                'ubar': ubar, 'u': u}

    def backward(self, deltay, fd, prev_fd, prev_gd):
        """Gated recurrent unit backward pass.

        Args:
            deltay (np.array): Gradients at t from the layer above.
            fd (dict): Forward dictionary recording the forward pass values.
            prev_fd (dict): Dictionary at time t+1.
            prev_gd (dict): Previous gradients at time t+1.

        Returns:
            A dict with:
                deltah: New recurrent gradients.
                deltaz: State candidate gradients.
                deltau: Update gate gradients.
                deltar: Reset gate gradients.
                dWout: Output projection weight matrix gradients.
                dbout: Output projection bias gadients.
                dW: State candidate input matrix gradients.
                dWu: Update gate input matrix gradients.
                dWr: Reset gate input matrix gradients.
                dV: State candidate recurrent matrix gradients.
                dVu: Update gate recurrent matrix gradients.
                dVr: Reset gate recurrent matrix gradients.
                db: State candidate bias gradients.
                dbu: Update gate bias gradients.
                dbr: Reset gate bias gradients.
        """
        # projection backward
        dWout = np.matmul(deltay, np.transpose(fd['h'], [0, 2, 1]))
        dbout = 1*deltay
        deltay = np.matmul(np.transpose(self.weights['Wout'], [0, 2, 1]),
                           deltay)

        # block backward
        wtdz = np.matmul(np.transpose(self.weights['W'], [0, 2, 1]),
                         prev_gd['deltaz'])
        deltah = deltay + (1 - fd['u'])*prev_gd['deltah'] \
            + fd['r']*wtdz \
            + np.matmul(np.transpose(self.weights['Wu'], [0, 2, 1]),
                        prev_gd['deltau']) \
            + np.matmul(np.transpose(self.weights['Wr'], [0, 2, 1]),
                        prev_gd['deltar'])

        deltaz = fd['u'] * deltah * self.state_activation.prime(fd['zbar'])
        deltau = (fd['z'] - fd['h']) * prev_gd['deltah'] \
            * self.gate_u_act.prime(fd['ubar'])
        deltar = fd['h']*wtdz*self.gate_r_act.prime(fd['rbar'])

        # weight backward
        dV = np.matmul(deltaz, np.transpose(fd['x'], [0, 2, 1]))
        dVu = np.matmul(deltau, np.transpose(fd['x'], [0, 2, 1]))
        dVr = np.matmul(deltar, np.transpose(fd['x'], [0, 2, 1]))

        dW = np.matmul(prev_gd['deltaz'], np.transpose(fd['h'], [0, 2, 1]))
        dWu = np.matmul(prev_gd['deltau'], np.transpose(fd['h'], [0, 2, 1]))
        dWr = np.matmul(prev_gd['deltar'], np.transpose(fd['h'], [0, 2, 1]))

        db = deltaz
        dbu = deltau
        dbr = deltar

        return {'deltah': deltah, 'dWout': dWout, 'dbout': dbout,
                'deltaz': deltaz, 'deltau': deltau, 'deltar': deltar,
                'dW': dW, 'dWu': dWu, 'dWr': dWr,
                'dV': dV, 'dVu': dVu, 'dVr': dVr,
                'db': db, 'dbu': dbu, 'dbr': dbr}
