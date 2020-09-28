# This script trains a basic elman rnn cell on the
# memory problem.

import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_memory
from numpy_cells import BasicCell, MSELoss

if __name__ == '__main__':
    n_train = int(9e5)
    n_test = int(1e4)
    baseline = 0.167
    time_steps = 12
    batch_size = 25
    lr = 0.1
    cell = BasicCell(hidden_size=56, input_size=1)
    cost = MSELoss()

    train_x, train_y = generate_data_memory(time_steps, n_train)

    train_x_lst = np.array_split(train_x, n_train//batch_size, axis=1)
    train_y_lst = np.array_split(train_y, n_train//batch_size, axis=0)

    iterations = len(train_x_lst)
    assert len(train_x_lst) == len(train_y_lst)

    h = cell.zero_state(batch_size)
    loss_lst = []
    # train cell
    for i in range(iterations):
        x = train_x_lst[i]
        y = train_y_lst[i]

        x = np.expand_dims(x, -1)
        y = np.expand_dims(y, -1)

        out_lst = []
        h_lst = []
        # forward
        # TODO: update
        for t in range(time_steps):
            out, h = cell.forward(x=x[t, :, :, :], h=h)
            out_lst.append(out)
            h_lst.append(h)
        loss = cost.forward(y, out_lst[-1])
        deltay = np.zeros((time_steps, batch_size, 1, 1))
        deltay[-1, :, :, :] = cost.backward(y, out_lst[-1])
        deltah = cell.zero_state(batch_size)

        grad_lst = []
        # backward
        for t in reversed(range(time_steps)):
            deltah, dWhh, dWxh, dbh, dWhy, dby = \
                cell.backward(deltay=deltay[t, :, :, :],
                              deltah=deltah,
                              x=x[t, :, :, :],
                              h=h_lst[t],
                              hm1=h_lst[t-1])
            grad_lst.append([dWhh, dWxh, dbh, dWhy, dby])
        ldWhh, ldWxh, ldbh, ldWhy, ldby = zip(*grad_lst)
        dWhh = np.stack(ldWhh, axis=0)
        dWxh = np.stack(ldWxh, axis=0)
        dbh = np.stack(ldbh, axis=0)
        dWhy = np.stack(ldWhy, axis=0)
        dby = np.stack(ldby, axis=0)
        # backprop in time requires us to sum the gradients at each
        # point in time.

        # todo add clipping.

        # update
        cell.Whh += -lr*np.expand_dims(np.mean(np.sum(dWhh, axis=0), 0), 0)
        cell.Wxh += -lr*np.expand_dims(np.mean(np.sum(dWxh, axis=0), 0), 0)
        cell.bh += -lr*np.expand_dims(np.mean(np.sum(dbh, axis=0), 0), 0)
        cell.Why += -lr*np.expand_dims(np.mean(np.sum(dWhy, axis=0), 0), 0)
        cell.by += -lr*np.expand_dims(np.mean(np.sum(dby, axis=0), 0), 0)

        if i % 10 == 0:
            print(i, 'loss', "%.4f" % loss, 'baseline', baseline,
                  'lr', "%.6f" % lr,
                  'done', "%.3f" % (i/iterations))
        loss_lst.append(loss)

        if i % 1000 == 0 and i > 0:
            lr = lr * 0.95

    print(y[:, 0, 0])
    print(out[:, 0, 0])
    plt.semilogy(loss_lst)
    plt.show()
