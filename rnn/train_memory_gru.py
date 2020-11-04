# Created by moritz (wolter@cs.uni-bonn.de)
# This script trains a long short term memory cell on the
# memory problem.
# TODO: fixme.

import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_memory
from numpy_cells import GRU, Sigmoid, CrossEntropyCost

if __name__ == '__main__':
    n_train = int(40e5)
    n_test = int(1e4)
    time_steps = 1
    output_size = 10
    n_sequence = 10
    train_data = generate_data_memory(time_steps, n_train, n_sequence)
    test_data = generate_data_memory(time_steps, n_test, n_sequence)
    # --- baseline ----------------------
    baseline = np.log(8) * 10/(time_steps + 20)  # TODO: FIXME!
    print("Baseline is " + str(baseline))
    batch_size = 100
    lr = .1
    cell = GRU(hidden_size=64, input_size=10, output_size=output_size)
    sigmoid = Sigmoid()

    cost = CrossEntropyCost()

    train_x, train_y = generate_data_memory(time_steps, n_train, n_sequence)
    train_x_lst = np.array_split(train_x, n_train//batch_size, axis=0)
    train_y_lst = np.array_split(train_y, n_train//batch_size, axis=0)

    iterations = len(train_x_lst)
    assert len(train_x_lst) == len(train_y_lst)

    # initialize cell state.
    fd0 = {'h': cell.zero_state(batch_size),
           'r': cell.zero_state(batch_size),
           'u': cell.zero_state(batch_size)}
    loss_lst = []
    acc_lst = []
    lr_lst = []
    # train cell
    for i in range(iterations):
        xx = train_x_lst[i]
        yy = train_y_lst[i]

        x_one_hot = np.zeros([batch_size, 20+time_steps, n_sequence])
        y_one_hot = np.zeros([batch_size, 20+time_steps, n_sequence])
        # one hote encode the inputs.
        for b in range(batch_size):
            for t in range(20+time_steps):
                x_one_hot[b, t, xx[b, t]] = 1
                y_one_hot[b, t, yy[b, t]] = 1

        x = np.expand_dims(x_one_hot, -1)
        y = np.expand_dims(y_one_hot, -1)

        out_lst = []
        fd_lst = []
        # forward
        fd = fd0
        for t in range(time_steps+20):
            fd = cell.forward(x=x[:, t, :, :],
                              h=fd['h'])
            fd_lst.append(fd)
            out = sigmoid.forward(fd['y'])
            out_lst.append(out)

        out_array = np.stack(out_lst, 1)
        loss = cost.forward(label=y[:, -10:, :, :],
                            out=out_array[:, -10:, :, :])
        deltay = np.zeros([batch_size, time_steps+20, n_sequence, 1])
        deltay[:, -10:, :, :] = cost.backward(label=y[:, -10:, :, :],
                                              out=out_array[:, -10:, :, :])

        gd = {'deltaz': cell.zero_state(batch_size),
              'deltah': cell.zero_state(batch_size),
              'deltau': cell.zero_state(batch_size),
              'deltar': cell.zero_state(batch_size)}

        # compute accuracy
        y_net = np.squeeze(np.argmax(out_array, axis=2))
        mem_net = y_net[:, -10:]
        mem_y = yy[:, -10:]
        acc = np.sum((mem_y == mem_net).astype(np.float32))
        acc = acc/(batch_size * 10.)
        # import pdb;pdb.set_trace()
        acc_lst.append(acc)

        gd_lst = []
        grad_lst = []
        # backward
        fd_lst.append(fd0)
        for t in reversed(range(time_steps+20)):
            gd = cell.backward(deltay=deltay[:, t, :, :],
                               fd=fd_lst[t],
                               next_fd=fd_lst[t+1],
                               prev_fd=fd_lst[t-1],
                               next_gd=gd)
            gd_lst.append(gd)
            grad_lst.append([gd['dWout'], gd['dbout'],
                             gd['dW'], gd['dWu'], gd['dWr'],
                             gd['dV'], gd['dVu'], gd['dVr'],
                             gd['db'], gd['dbu'], gd['dbr']])
        ldWout, ldbout, ldW, ldWu, ldWr, ldV, ldVu,\
            ldVr, ldb, ldbu, ldbr = zip(*grad_lst)
        dWout = np.stack(ldWout, axis=0)
        dbout = np.stack(ldbout, axis=0)
        dW = np.stack(ldW, axis=0)
        dWu = np.stack(ldWu, axis=0)
        dWr = np.stack(ldWr, axis=0)
        dV = np.stack(ldV, axis=0)
        dVu = np.stack(ldVu, axis=0)
        dVr = np.stack(ldVr, axis=0)
        db = np.stack(ldb, axis=0)
        dbu = np.stack(ldbu, axis=0)
        dbr = np.stack(ldbr, axis=0)

        # backprop in time requires us to sum the gradients at each
        # point in time. Clip between -1 and 1.
        dWout = np.clip(np.sum(ldWout, axis=0), -1, 1)
        dbout = np.clip(np.sum(ldbout, axis=0), -1, 1)
        dW = np.clip(np.sum(ldW, axis=0), -1, 1)
        dWu = np.clip(np.sum(ldWu, axis=0), -1, 1)
        dWr = np.clip(np.sum(ldWr, axis=0), -1, 1)
        dV = np.clip(np.sum(ldV, axis=0), -1, 1)
        dVu = np.clip(np.sum(ldVu, axis=0), -1, 1)
        dVr = np.clip(np.sum(ldVr, axis=0), -1, 1)
        db = np.clip(np.sum(ldb, axis=0), -1, 1)
        dbu = np.clip(np.sum(ldbu, axis=0), -1, 1)
        dbr = np.clip(np.sum(ldbr, axis=0), -1, 1)

        # update
        cell.weights['Wout'] += -lr*np.expand_dims(np.mean(dWout, 0), 0)
        cell.weights['bout'] += -lr*np.expand_dims(np.mean(dbout, 0), 0)
        cell.weights['W'] += -lr*np.expand_dims(np.mean(dW, 0), 0)
        cell.weights['Wu'] += -lr*np.expand_dims(np.mean(dWu, 0), 0)
        cell.weights['Wr'] += -lr*np.expand_dims(np.mean(dWr, 0), 0)
        cell.weights['V'] += -lr*np.expand_dims(np.mean(dV, 0), 0)
        cell.weights['Vu'] += -lr*np.expand_dims(np.mean(dVu, 0), 0)
        cell.weights['Vr'] += -lr*np.expand_dims(np.mean(dVr, 0), 0)
        cell.weights['b'] += -lr*np.expand_dims(np.mean(db, 0), 0)
        cell.weights['bu'] += -lr*np.expand_dims(np.mean(dbu, 0), 0)
        cell.weights['br'] += -lr*np.expand_dims(np.mean(dbr, 0), 0)

        if i % 10 == 0:
            print(i, 'loss', "%.4f" % loss, 'baseline', "%.4f" % baseline,
                  'acc', "%.4f" % acc, 'lr', "%.6f" % lr,
                  'done', "%.3f" % (i/iterations))
        loss_lst.append(loss)

        if i % 500 == 0 and i > 0:
            lr = lr * 0.96

            # import pdb;pdb.set_trace()
            print('net', y_net[0, -10:])
            print('gt ', yy[0, -10:])

    print('net', y_net[0, -10:])
    print('gt ', yy[0, -10:])
    plt.semilogy(loss_lst)
    plt.title('memory gru loss')
    plt.xlabel('weight updates')
    plt.ylabel('cross entropy')
    plt.show()

    plt.plot(acc_lst)
    plt.show()
