import numpy as np
import matplotlib.pyplot as plt

from generate_adding_memory import generate_data_adding
from numpy_cells import LSTMcell, MSELoss

if __name__ == '__main__':
    n_train = int(9e5)
    n_test = int(1e4)
    baseline = 0.167
    time_steps = 12
    batch_size = 25
    lr = 0.1
    cell = LSTMcell(hidden_size=56, input_size=2)
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
        ibar_lst = []
        fbar_lst = []
        obar_lst = []
        # forward
        for t in range(time_steps):
            out, c, h, zbar, ibar, fbar, obar = \
                cell.forward(x=x[t, :, :, :], c=c, h=h)
            out_lst.append(out)
            c_lst.append(c)
            h_lst.append(h)
            zbar_lst.append(zbar)
            ibar_lst.append(ibar)
            fbar_lst.append(fbar)
            obar_lst.append(obar)
        loss = cost.forward(y, out_lst[-1])
        deltay = np.zeros((time_steps, batch_size, 1, 1))
        deltay[-1, :, :, :] = cost.backward(y, out_lst[-1])
        deltah = cell.zero_state(batch_size)
        deltac = cell.zero_state(batch_size)
        deltao = cell.zero_state(batch_size)
        deltai = cell.zero_state(batch_size)
        deltaf = cell.zero_state(batch_size)

        grad_lst = []
        # backward
        for t in reversed(range(time_steps)):
            deltac, deltao, deltai, deltaf, \
                dWout, dbout, dWz, dWi, dWf, dWo, dRz, dRi,\
                dRf, dRo, dbz, dbi, dbf, dbo,\
                dpi, dpf, dpo = \
                cell.backward(deltay=deltay[t, :, :, :],
                              deltac=deltac,
                              deltao=deltao,
                              deltai=deltai,
                              deltaf=deltaf,
                              x=x[t, :, :, :],
                              c=c_lst[t],
                              h=h_lst[t],
                              cm1=c_lst[t-1],
                              zbar=zbar_lst[t],
                              obar=obar_lst[t],
                              ibar=ibar_lst[t],
                              fbar=fbar_lst[t])
            grad_lst.append([dWout, dbout, dWz, dWi, dWf,
                             dWo, dRz, dRi, dRf, dRo, dbz,
                             dbi, dbf, dbo, dpi, dpf, dpo])
        ldWout, ldbout, ldWz, ldWi, ldWf, ldWo, ldRz, ldRi,\
            ldRf, ldRo, ldbz, ldbi, ldbf, ldbo,\
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

        # todo add clipping.

        # update
        cell.Wout += -lr*np.expand_dims(np.mean(np.sum(dWout, axis=0), 0), 0)
        cell.bout += -lr*np.expand_dims(np.mean(np.sum(dbout, axis=0), 0), 0)
        cell.Wz += -lr*np.expand_dims(np.mean(np.sum(dWz, axis=0), 0), 0)
        cell.Wi += -lr*np.expand_dims(np.mean(np.sum(dWi, axis=0), 0), 0)
        cell.Wf += -lr*np.expand_dims(np.mean(np.sum(dWf, axis=0), 0), 0)
        cell.Wo += -lr*np.expand_dims(np.mean(np.sum(dWo, axis=0), 0), 0)
        cell.Rz += -lr*np.expand_dims(np.mean(np.sum(dRz, axis=0), 0), 0)
        cell.Ri += -lr*np.expand_dims(np.mean(np.sum(dRi, axis=0), 0), 0)
        cell.Rf += -lr*np.expand_dims(np.mean(np.sum(dRf, axis=0), 0), 0)
        cell.Ro += -lr*np.expand_dims(np.mean(np.sum(dRo, axis=0), 0), 0)
        cell.bz += -lr*np.expand_dims(np.mean(np.sum(dbz, axis=0), 0), 0)
        cell.bi += -lr*np.expand_dims(np.mean(np.sum(dbi, axis=0), 0), 0)
        cell.bf += -lr*np.expand_dims(np.mean(np.sum(dbf, axis=0), 0), 0)
        cell.bo += -lr*np.expand_dims(np.mean(np.sum(dbo, axis=0), 0), 0)
        cell.pi += -lr*np.expand_dims(np.mean(np.sum(dpi, axis=0), 0), 0)
        cell.pf += -lr*np.expand_dims(np.mean(np.sum(dpf, axis=0), 0), 0)
        cell.po += -lr*np.expand_dims(np.mean(np.sum(dpo, axis=0), 0), 0)
        
        if i % 10 == 0:
            print(i, 'loss', "%.4f" % loss, 'baseline', baseline,
                  'lr', "%.6f" % lr,
                  'done', "%.3f" % (i/iterations))
        loss_lst.append(loss)

        if i % 1000 == 0 and i > 0:
            lr = lr * 0.90

    print(y[:, 0, 0])
    print(out[:, 0, 0])
    plt.semilogy(loss_lst)
    plt.show()

    # test
    test_x, test_y = generate_data_adding(time_steps, n_test)
