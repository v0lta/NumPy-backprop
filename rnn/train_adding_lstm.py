# Created by moritz (wolter@cs.uni-bonn.de)
# This script trains a LSTM cell on the adding problem using numpy only.

import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_adding
from numpy_cells import LSTMcell, MSELoss

if __name__ == '__main__':
    n_train = int(10e5)
    n_test = int(1e4)
    baseline = 0.167
    time_steps = 50
    batch_size = 100
    lr = 1.0
    cell = LSTMcell(hidden_size=64, input_size=2)
    cost = MSELoss()

    train_x, train_y = generate_data_adding(time_steps, n_train)

    train_x_lst = np.array_split(train_x, n_train//batch_size, axis=1)
    train_y_lst = np.array_split(train_y, n_train//batch_size, axis=0)

    iterations = len(train_x_lst)
    assert len(train_x_lst) == len(train_y_lst)

    # initialize cell state.
    fd0 = {'c': cell.zero_state(batch_size),
           'h': cell.zero_state(batch_size),
           'f': cell.zero_state(batch_size)}
    loss_lst = []
    lr_lst = []

    # train cell
    for i in range(iterations):
        x = train_x_lst[i]
        y = train_y_lst[i]

        x = np.expand_dims(x, -1)
        y = np.expand_dims(y, -1)

        fd_lst = []
        # forward
        fd = fd0
        for t in range(time_steps):
            fd = cell.forward(x=x[t, :, :, :],
                              c=fd['c'], h=fd['h'])
            fd_lst.append(fd)

        loss = cost.forward(y, fd_lst[-1]['y'])
        deltay = np.zeros((time_steps, batch_size, 1, 1))
        deltay[-1, :, :, :] = cost.backward(y, fd_lst[-1]['y'])

        gd = {'deltah': cell.zero_state(batch_size),
              'deltac': cell.zero_state(batch_size),
              'deltaz': cell.zero_state(batch_size),
              'deltao': cell.zero_state(batch_size),
              'deltai': cell.zero_state(batch_size),
              'deltaf': cell.zero_state(batch_size)}
        gd_lst = []
        grad_lst = []
        # backward
        fd_lst.append(fd0)
        for t in reversed(range(time_steps)):
            gd = cell.backward(deltay=deltay[t, :, :, :],
                               fd=fd_lst[t],
                               next_fd=fd_lst[t+1],
                               prev_fd=fd_lst[t-1],
                               next_gd=gd)
            gd_lst.append(gd)
            # TODO: Move elsewhere.
            grad_lst.append([gd['dWout'], gd['dbout'],
                             gd['dWz'], gd['dWi'], gd['dWf'], gd['dWo'],
                             gd['dRz'], gd['dRi'], gd['dRf'], gd['dRo'],
                             gd['dbz'], gd['dbi'], gd['dbf'], gd['dbo'],
                             gd['dpi'], gd['dpf'], gd['dpo']])
        ldWout, ldbout,\
            ldWz, ldWi, ldWf, ldWo, \
            ldRz, ldRi, ldRf, ldRo, \
            ldbz, ldbi, ldbf, ldbo,\
            ldpi, ldpf, ldpo = zip(*grad_lst)
        dWout = np.stack(ldWout, axis=0)
        dbout = np.stack(ldbout, axis=0)
        dWz = np.stack(ldWz, axis=0)
        dWi = np.stack(ldWi, axis=0)
        dWf = np.stack(ldWf, axis=0)
        dWo = np.stack(ldWo, axis=0)
        dRz = np.stack(ldRz, axis=0)
        dRi = np.stack(ldRi, axis=0)
        dRf = np.stack(ldRf, axis=0)
        dRo = np.stack(ldRo, axis=0)
        dbz = np.stack(ldbz, axis=0)
        dbi = np.stack(ldbi, axis=0)
        dbf = np.stack(ldbf, axis=0)
        dbo = np.stack(ldbo, axis=0)
        dpi = np.stack(ldpi, axis=0)
        dpf = np.stack(ldpf, axis=0)
        dpo = np.stack(ldpo, axis=0)

        # backprop in time requires us to sum the gradients at each
        # point in time.
        # clipping prevents gradient explosion.
        dWout = np.clip(np.sum(dWout, axis=0), -1, 1)
        dbout = np.clip(np.sum(dbout, axis=0), -1, 1)
        dWz = np.clip(np.sum(dWz, axis=0), -1, 1)
        dWi = np.clip(np.sum(dWi, axis=0), -1, 1)
        dWf = np.clip(np.sum(dWf, axis=0), -1, 1)
        dWo = np.clip(np.sum(dWo, axis=0), -1, 1)
        dRz = np.clip(np.sum(dRz, axis=0), -1, 1)
        dRi = np.clip(np.sum(dRi, axis=0), -1, 1)
        dRf = np.clip(np.sum(dRf, axis=0), -1, 1)
        dRo = np.clip(np.sum(dRo, axis=0), -1, 1)
        dbz = np.clip(np.sum(dbz, axis=0), -1, 1)
        dbi = np.clip(np.sum(dbi, axis=0), -1, 1)
        dbf = np.clip(np.sum(dbf, axis=0), -1, 1)
        dbo = np.clip(np.sum(dbo, axis=0), -1, 1)
        dpi = np.clip(np.sum(dpi, axis=0), -1, 1)
        dpf = np.clip(np.sum(dpf, axis=0), -1, 1)
        dpo = np.clip(np.sum(dpo, axis=0), -1, 1)

        # update
        cell.weights['Wout'] += -lr*np.expand_dims(np.mean(dWout, 0), 0)
        cell.weights['bout'] += -lr*np.expand_dims(np.mean(dbout, 0), 0)
        cell.weights['Wz'] += -lr*np.expand_dims(np.mean(dWz, 0), 0)
        cell.weights['Wi'] += -lr*np.expand_dims(np.mean(dWi, 0), 0)
        cell.weights['Wf'] += -lr*np.expand_dims(np.mean(dWf, 0), 0)
        cell.weights['Wo'] += -lr*np.expand_dims(np.mean(dWo, 0), 0)
        cell.weights['Rz'] += -lr*np.expand_dims(np.mean(dRz, 0), 0)
        cell.weights['Ri'] += -lr*np.expand_dims(np.mean(dRi, 0), 0)
        cell.weights['Rf'] += -lr*np.expand_dims(np.mean(dRf, 0), 0)
        cell.weights['Ro'] += -lr*np.expand_dims(np.mean(dRo, 0), 0)
        cell.weights['bz'] += -lr*np.expand_dims(np.mean(dbz, 0), 0)
        cell.weights['bi'] += -lr*np.expand_dims(np.mean(dbi, 0), 0)
        cell.weights['bf'] += -lr*np.expand_dims(np.mean(dbf, 0), 0)
        cell.weights['bo'] += -lr*np.expand_dims(np.mean(dbo, 0), 0)
        cell.weights['pi'] += -lr*np.expand_dims(np.mean(dpi, 0), 0)
        cell.weights['pf'] += -lr*np.expand_dims(np.mean(dpf, 0), 0)
        cell.weights['po'] += -lr*np.expand_dims(np.mean(dpo, 0), 0)

        if i % 10 == 0:
            print(i, 'mse loss', "%.4f" % loss, 'baseline', baseline,
                  'lr', "%.6f" % lr,
                  'done', "%.3f" % (i/iterations))
        loss_lst.append(loss)

        if i % 500 == 0 and i > 0:
            lr = lr * 0.95
        lr_lst.append(lr)

    # 0th batch marked inputs
    print(x[x[:, 0, 1, 0] == 1., 0, 0, 0])
    # desired output for all batches
    print(y[:10, 0, 0])
    # network output for all batches
    print(fd['y'][:10, 0, 0])
    plt.semilogy(loss_lst)
    plt.title('loss adding problem lstm')
    plt.xlabel('weight updates')
    plt.ylabel('mean squared error')
    plt.show()
