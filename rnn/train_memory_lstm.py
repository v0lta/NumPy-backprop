# Created by moritz (wolter@cs.uni-bonn.de)
# This script trains a long short term memory cell on the
# memory problem.
# TODO: fixme.

import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_memory
from numpy_cells import LSTMcell, Sigmoid, CrossEntropyCost
from opt import RMSprop

if __name__ == '__main__':
    n_train = int(40e5)
    n_test = int(1e4)
    time_steps = 1
    output_size = 10
    n_sequence = 10
    train_data = generate_data_memory(time_steps, n_train, n_sequence)
    test_data = generate_data_memory(time_steps, n_test, n_sequence)
    # --- baseline ----------------------
    # baseline = np.log(8) * 10/(time_steps + 20)
    # print("Baseline is " + str(baseline))  # TODO: Fixme!
    batch_size = 100
    lr = 0.01
    cell = LSTMcell(hidden_size=64, input_size=10, output_size=output_size)
    opt = RMSprop(lr=lr)
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
        loss = cost.forward(label=y[:, :, :, :],
                            out=out_array[:, :, :, :])
        deltay = np.zeros([batch_size, time_steps+20, n_sequence, 1])
        deltay = cost.backward(label=y[:, :, :, :],
                               out=out_array[:, :, :, :])

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
        grad_double_list = list(zip(*grad_lst))

        # stack the gradients in time.
        stack_list = []
        for grad_list in grad_double_list:
            stack_list.append(np.stack(grad_list))

        # backprop in time requires us to sum the gradients at each
        # point in time. Clip between -1 and 1.
        clip_list = []
        for grads in stack_list:
            # clip_list.append(np.clip(np.sum(grads, axis=0), -1, 1))
            clip_list.append(np.sum(grads, axis=0))

        # turn back into dict for optimizer.
        keys = ['Wout', 'bout',
                'Wz', 'Wi', 'Wf', 'Wo',
                'Rz', 'Ri', 'Rf', 'Ro',
                'bz', 'bi', 'bf', 'bo',
                'pi', 'pf', 'po']
        gd = {}
        for key_no, key in enumerate(keys):
            gd[key] = clip_list[key_no]

        # update
        opt.step(cell, gd)

        if i % 10 == 0:
            print(i, 'loss', "%.4f" % loss,
                  'acc', "%.4f" % acc, 'lr', "%.6f" % opt.lr,
                  'done', "%.3f" % (i/iterations))
        loss_lst.append(loss)

        if i % 500 == 0 and i > 0:
            opt.lr = opt.lr * 1.

            # import pdb;pdb.set_trace()
            print('net', y_net[0, :])
            print('gt ', yy[0, :])

    print('net', y_net[0, -10:])
    print('gt ', yy[0, -10:])
    plt.semilogy(loss_lst)
    plt.title('memory lstm loss')
    plt.xlabel('weight updates')
    plt.ylabel('cross entropy')
    plt.show()

    plt.plot(acc_lst)
    plt.show()
