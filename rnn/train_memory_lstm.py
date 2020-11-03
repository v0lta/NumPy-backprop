# Created by moritz (wolter@cs.uni-bonn.de)
# This script trains a long short term memory cell on the
# memory problem.
# TODO: fixme.

import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_memory
from numpy_cells import LSTMcell, Sigmoid, CrossEntropyCost

if __name__ == '__main__':
    n_train = int(40e5)
    n_test = int(1e4)
    time_steps = 30
    output_size = 10
    n_sequence = 10
    train_data = generate_data_memory(time_steps, n_train, n_sequence)
    test_data = generate_data_memory(time_steps, n_test, n_sequence)
    # --- baseline ----------------------
    baseline = np.log(8) * 10/(time_steps + 20)
    print("Baseline is " + str(baseline))
    batch_size = 100
    lr = 1.0
    cell = LSTMcell(hidden_size=64, input_size=10, output_size=output_size)
    sigmoid = Sigmoid()

    cost = CrossEntropyCost()

    train_x, train_y = generate_data_memory(time_steps, n_train, n_sequence)
    train_x_lst = np.array_split(train_x, n_train//batch_size, axis=0)
    train_y_lst = np.array_split(train_y, n_train//batch_size, axis=0)

    iterations = len(train_x_lst)
    assert len(train_x_lst) == len(train_y_lst)

    # initialize cell state.
    fd0 = {'c': cell.zero_state(batch_size),
           'h': cell.zero_state(batch_size),
           'f': cell.zero_state(batch_size)}
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
                              c=fd['c'], h=fd['h'])
            fd_lst.append(fd)
            out = sigmoid.forward(fd['y'])
            out_lst.append(out)

        out_array = np.stack(out_lst, 1)
        loss = cost.forward(label=y[:, -10:, :, :],
                            out=out_array[:, -10:, :, :])
        deltay = np.zeros([batch_size, time_steps+20, n_sequence, 1])
        deltay[:, -10:, :, :] = cost.backward(label=y[:, -10:, :, :],
                                              out=out_array[:, -10:, :, :])

        gd = {'deltah': cell.zero_state(batch_size),
              'deltac': cell.zero_state(batch_size),
              'deltaz': cell.zero_state(batch_size),
              'deltao': cell.zero_state(batch_size),
              'deltai': cell.zero_state(batch_size),
              'deltaf': cell.zero_state(batch_size)}

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
            # TODO: Move elsewhere.
            grad_lst.append([gd['dWout'], gd['dbout'],
                             gd['dWz'], gd['dWi'], gd['dWf'], gd['dWo'],
                             gd['dRz'], gd['dRi'], gd['dRf'], gd['dRo'],
                             gd['dbz'], gd['dbi'], gd['dbf'], gd['dbo'],
                             gd['dpi'], gd['dpf'], gd['dpo']])
        ldWout, ldbout, \
            ldWz, ldWi, ldWf, ldWo,\
            ldRz, ldRi, ldRf, ldRo,\
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
        dWout = np.clip(np.sum(dWout, axis=0), -1.0, 1.0)
        dbout = np.clip(np.sum(dbout, axis=0), -1.0, 1.0)
        dWz = np.clip(np.sum(dWz, axis=0), -1.0, 1.0)
        dWi = np.clip(np.sum(dWi, axis=0), -1.0, 1.0)
        dWf = np.clip(np.sum(dWf, axis=0), -1.0, 1.0)
        dWo = np.clip(np.sum(dWo, axis=0), -1.0, 1.0)
        dRz = np.clip(np.sum(dRz, axis=0), -1.0, 1.0)
        dRi = np.clip(np.sum(dRi, axis=0), -1.0, 1.0)
        dRf = np.clip(np.sum(dRf, axis=0), -1.0, 1.0)
        dRo = np.clip(np.sum(dRo, axis=0), -1.0, 1.0)
        dbz = np.clip(np.sum(dbz, axis=0), -1.0, 1.0)
        dbi = np.clip(np.sum(dbi, axis=0), -1.0, 1.0)
        dbf = np.clip(np.sum(dbf, axis=0), -1.0, 1.0)
        dbo = np.clip(np.sum(dbo, axis=0), -1.0, 1.0)
        dpi = np.clip(np.sum(dpi, axis=0), -1.0, 1.0)
        dpf = np.clip(np.sum(dpf, axis=0), -1.0, 1.0)
        dpo = np.clip(np.sum(dpo, axis=0), -1.0, 1.0)

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
            print(i, 'loss', "%.4f" % loss, 'baseline', "%.4f" % baseline,
                  'acc', "%.4f" % acc, 'lr', "%.6f" % lr,
                  'done', "%.3f" % (i/iterations))
        loss_lst.append(loss)

        if i % 500 == 0 and i > 0:
            lr = lr * 0.95

            # import pdb;pdb.set_trace()
            print('net', y_net[0, -10:])
            print('gt ', yy[0, -10:])

    print('net', y_net[0, -10:])
    print('gt ', yy[0, -10:])
    plt.semilogy(loss_lst)
    plt.title('memory lstm loss')
    plt.xlabel('weight updates')
    plt.ylabel('cross entropy')
    plt.show()

    plt.plot(acc_lst)
    plt.show()
