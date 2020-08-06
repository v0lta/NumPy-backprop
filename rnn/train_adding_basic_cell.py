import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_adding
from numpy_cells import BasicCell, MSELoss

if __name__ == '__main__':
    n_train = int(9e5)
    n_test = int(1e4)
    baseline = 0.167
    time_steps = 5
    batch_size = 50
    lr = .05
    cell = BasicCell(hidden_size=250, input_size=2)
    cost = MSELoss()

    train_x, train_y = generate_data_adding(time_steps, n_train)

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
        for t in range(time_steps):
            out, h = cell.forward(x=x[t, :, :, :], h=h)
            out_lst.append(out)
            h_lst.append(h)
        loss = cost.forward(y, out_lst[-1])
        deltay = cost.backward(y, out_lst[-1])
        deltah = cell.zero_state(batch_size)

        grad_lst = []
        # backward
        for t in reversed(range(time_steps)):
            deltah, dWhh, dWxh, dbh, dWhy, dby = \
                cell.backward(deltay=deltay, deltah=deltah,
                              x=x[t, :, :, :], h=h_lst[t], y=out_lst[t])
            grad_lst.append([dWhh, dWxh, dbh, dWhy, dby])
        ldWhh, ldWxh, ldbh, ldWhy, ldby = zip(*grad_lst)
        dWhh = np.stack(ldWhh)
        dWxh = np.stack(ldWxh)
        dbh = np.stack(ldbh)
        dWhy = np.stack(ldWhy)
        dby = np.stack(ldby)
        # backprop in time requires us to sum the gradients at each
        # point in time.

        # update
        cell.Whh = -lr*np.expand_dims(np.mean(np.sum(dWhh, axis=0), axis=0), 0)
        cell.Wxh = -lr*np.expand_dims(np.mean(np.sum(dWxh, axis=0), axis=0), 0)
        cell.bh = -lr*np.expand_dims(np.mean(np.sum(dbh, axis=0), axis=0), 0)
        cell.Why = -lr*np.expand_dims(np.mean(np.sum(dWhy, axis=0), axis=0), 0)
        cell.by = -lr*np.expand_dims(np.mean(np.sum(dby, axis=0), axis=0), 0)

        if i % 5 == 0:
            print(i, loss, baseline)
        loss_lst.append(loss)

    plt.semilogy(loss_lst)
    plt.show()

    # test
    test_x, test_y = generate_data_adding(time_steps, n_test)
