import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_adding
from numpy_cells import GRU, MSELoss

if __name__ == '__main__':
    n_train = int(9e5)
    n_test = int(1e4)
    baseline = 0.167
    time_steps = 12
    batch_size = 25
    lr = 0.1
    cell = GRU(hidden_size=56, input_size=2)
    cost = MSELoss()

    train_x, train_y = generate_data_adding(time_steps, n_train)

    train_x_lst = np.array_split(train_x, n_train//batch_size, axis=1)
    train_y_lst = np.array_split(train_y, n_train//batch_size, axis=0)

    iterations = len(train_x_lst)
    assert len(train_x_lst) == len(train_y_lst)

    # initialize cell state.
    c = cell.zero_state(batch_size)
    h = cell.zero_state(batch_size)
    loss_lst = []
    # train cell
    for i in range(iterations):
        x = train_x_lst[i]
        y = train_y_lst[i]

        x = np.expand_dims(x, -1)
        y = np.expand_dims(y, -1)

        out_lst = []
        c_lst = []
        h_lst = []
        zbar_lst = []
        hbar_lst = []
        ubar_lst = []
        rbar_lst = []

        # forward
        for t in range(time_steps):
            out, h, zbar, hbar, rbar, ubar = \
                cell.forward(x=x[t, :, :, :], h=h)
            out_lst.append(out)
            c_lst.append(c)
            h_lst.append(h)
            zbar_lst.append(zbar)
            hbar_lst.append(hbar)
            rbar_lst.append(rbar)
            ubar_lst.append(ubar)
        loss = cost.forward(y, out_lst[-1])
        deltay = np.zeros((time_steps, batch_size, 1, 1))
        deltay[-1, :, :, :] = cost.backward(y, out_lst[-1])
        deltaz = cell.zero_state(batch_size)
        deltah = cell.zero_state(batch_size)
        deltau = cell.zero_state(batch_size)
        deltar = cell.zero_state(batch_size)

        grad_lst = []
        # backward
        for t in reversed(range(time_steps)):
            deltah, deltaz, deltau, deltar, \
                dWout, dbout, dW, dWu, dWr, dV, dVu,\
                dVr, db, dbu, dbr = \
                cell.backward(deltay=deltay[t, :, :, :],
                              deltaz=deltaz,
                              deltah=deltah,
                              deltau=deltau,
                              deltar=deltar,
                              x=x[t, :, :, :],
                              h=h_lst[t],
                              hm1=h_lst[t-1],
                              zbar=zbar_lst[t],
                              ubar=ubar_lst[t],
                              rbar=rbar_lst[t])
            grad_lst.append([dWout, dbout, dW, dWu, dWr, dV, dVu,
                             dVr, db, dbu, dbr])
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
        cell.Wout += -lr*np.expand_dims(np.mean(dWout, 0), 0)
        cell.bout += -lr*np.expand_dims(np.mean(dbout, 0), 0)
        cell.W += -lr*np.expand_dims(np.mean(dW, 0), 0)
        cell.Wu += -lr*np.expand_dims(np.mean(dWu, 0), 0)
        cell.Wr += -lr*np.expand_dims(np.mean(dWr, 0), 0)
        cell.V += -lr*np.expand_dims(np.mean(dV, 0), 0)
        cell.Vu += -lr*np.expand_dims(np.mean(dVu, 0), 0)
        cell.Vr += -lr*np.expand_dims(np.mean(dVr, 0), 0)
        cell.b += -lr*np.expand_dims(np.mean(db, 0), 0)
        cell.bu += -lr*np.expand_dims(np.mean(dbu, 0), 0)
        cell.br += -lr*np.expand_dims(np.mean(dbr, 0), 0)

        if i % 10 == 0:
            print(i, 'loss', "%.4f" % loss, 'baseline', baseline,
                  'lr', "%.6f" % lr,
                  'done', "%.3f" % (i/iterations))
        loss_lst.append(loss)

        if i % 1000 == 0 and i > 0:
            lr = lr * 0.90

    # 0th batch marked inputs
    print(x[x[:, 0, 1, 0] == 1., 0, 0, 0])
    # desired output for all batches
    print(y[:, 0, 0])
    # network output for all batches
    print(out[:, 0, 0])
    plt.semilogy(loss_lst)
    plt.title('loss')
    plt.xlabel('weight updates')
    plt.ylabel('mean squared error')
    plt.show()

    # test
    test_x, test_y = generate_data_adding(time_steps, n_test)
