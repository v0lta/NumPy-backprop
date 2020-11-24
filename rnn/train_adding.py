# Created by moritz (wolter@cs.uni-bonn.de)
# This script trains a LSTM cell on the adding problem using numpy only.

import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_adding
from numpy_cells import LSTMcell, GRU, BasicCell
from numpy_cells import Sigmoid, MSELoss
from opt import RMSprop

if __name__ == '__main__':
    n_train = int(10e5)
    n_test = int(1e4)
    baseline = 0.167
    time_steps = 20
    batch_size = 100
    lr = 0.01
    cell = LSTMcell(hidden_size=64, input_size=2)
    # cell = GRU(hidden_size=64, input_size=2)
    # cell = BasicCell(hidden_size=64, input_size=2)
    cost = MSELoss()
    opt = RMSprop(lr=lr)

    print('Adding experiment started using:', type(cell), type(opt), lr)

    train_x, train_y = generate_data_adding(time_steps, n_train)

    train_x_lst = np.array_split(train_x, n_train//batch_size, axis=1)
    train_y_lst = np.array_split(train_y, n_train//batch_size, axis=0)

    iterations = len(train_x_lst)
    assert len(train_x_lst) == len(train_y_lst)

    # initialize cell state.
    fd0 = cell.zero_forward_dict(batch_size)
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

        gd = cell.zero_gradient_dict(batch_size)
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
            # get the weight related gradients at the
            # current time step.
            current_grad_list = []
            for key in cell.weight_keys:
                grad_key = cell.grad_to_weight_dict()[key]
                current_grad_list.append(gd[grad_key])

            grad_lst.append(current_grad_list)

        # extract the weight gradients for each weight
        # as a list.
        grad_double_list = list(zip(*grad_lst))

        # stack the weight gradients into an array
        # with time at the zeroth axis.
        stack_list = []
        for grad_list in grad_double_list:
            stack_list.append(np.stack(grad_list))

        # backprop in time requires us to sum the gradients at each
        # point in time. Clip between -1 and 1.
        clip_list = []
        for grads in stack_list:
            clip_list.append(np.clip(np.sum(grads, axis=0), -1, 1))
            # clip_list.append(np.sum(grads, axis=0))

        # turn back into dict for optimizer.
        gd = {}
        for key_no, key in enumerate(cell.weight_keys):
            gd[key] = clip_list[key_no]

        # update
        opt.step(cell, gd)

        if i % 10 == 0:
            print(i, 'mse loss', "%.4f" % loss, 'baseline', baseline,
                  'lr', "%.6f" % lr,
                  'done', "%.3f" % (i/iterations))
        loss_lst.append(loss)

        if i % 500 == 0 and i > 0:
            opt.lr = opt.lr * 1.
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
