# Created by moritz (wolter@cs.uni-bonn.de)
import struct
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy_layer import normalize
from numpy_layer import DenseLayer
from numpy_layer import MSELoss
from numpy_layer import CrossEntropyCost
from numpy_layer import ReLu
from numpy_layer import Sigmoid


def get_test_data():
    with open('./data/t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        img_data_test = data.reshape((size, nrows, ncols))

    with open('./data/t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        lbl_data_test = np.fromfile(f, dtype=np.dtype(np.uint8))
    return img_data_test, lbl_data_test


def get_train_data():
    with open('./data/train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        img_data_train = data.reshape((size, nrows, ncols))

    with open('./data/train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.fromfile(f, dtype=np.dtype(np.uint8))
    return img_data_train, lbl_data_train


if __name__ == '__main__':
    img_data_train, lbl_data_train = get_train_data()
    img_data_train, mean, std = normalize(img_data_train)

    lr = 1.0
    batch_size = 100
    dense = DenseLayer(784, 256)
    act1 = Sigmoid()
    dense2 = DenseLayer(256, 128)
    act2 = Sigmoid()
    dense3 = DenseLayer(128, 10)
    act3 = Sigmoid()
    cost = CrossEntropyCost()
    iterations = 5
    loss_lst = []
    acc_lst = []

    for e in range(iterations):
        shuffler = np.random.permutation(len(img_data_train))
        img_data_train = img_data_train[shuffler]
        lbl_data_train = lbl_data_train[shuffler]

        img_batches = np.split(img_data_train,
                               img_data_train.shape[0]//batch_size,
                               axis=0)
        label_batches = np.split(lbl_data_train,
                                 lbl_data_train.shape[0]//batch_size,
                                 axis=0)

        # for b in range(img_data_train.shape[0]):
        #    x = img_data_train[b].flatten()
        #    label = lbl_data_train[b]
        for no, img_batch in enumerate(img_batches):
            img_batch = np.reshape(img_batch, [img_batch.shape[0], -1])
            img_batch = np.expand_dims(img_batch, -1)
            labels = label_batches[no]
            y = []
            for b in range(batch_size):
                one_hot = np.zeros([10])
                one_hot[labels[b]] = 1
                y.append(one_hot)
            y = np.expand_dims(np.stack(y), -1)
            x = img_batch

            # forward pass
            h1 = dense.forward(x)
            h1_nl = act1.forward(h1)
            h2 = dense2.forward(h1_nl)
            h2_nl = act2.forward(h2)
            h3 = dense3.forward(h2_nl)
            y_hat = act3.forward(h3)
            loss = cost.forward(label=y, out=y_hat)

            # backward pass
            dl = cost.backward(label=y, out=y_hat)
            grads_dense3 = dense3.backward(inputs=h2_nl, delta=dl)
            dx3 = act2.backward(h2, grads_dense3['x'])
            grads_dense2 = dense2.backward(inputs=h1_nl, delta=dx3)
            dx2 = act1.backward(h1, grads_dense2['x'])
            grads_dense = dense.backward(inputs=x, delta=dx2)

            # update
            dense.weights['W'] += -lr*np.mean(grads_dense['W'], axis=0)
            dense.weights['b'] += -lr*np.mean(grads_dense['b'], axis=0)
            dense2.weights['W'] += -lr*np.mean(grads_dense2['W'], axis=0)
            dense2.weights['b'] += -lr*np.mean(grads_dense2['b'], axis=0)
            dense3.weights['W'] += -lr*np.mean(grads_dense3['W'], axis=0)
            dense3.weights['b'] += -lr*np.mean(grads_dense3['b'], axis=0)
            loss_lst.append(loss)

            true = np.sum((labels == np.squeeze(np.argmax(y_hat, axis=1))
                           ).astype(np.float32))
            acc = true/batch_size
            acc_lst.append(acc)
            if no % 5 == 0:
                print('e', e, 'b', no, 'loss', loss, 'acc', acc, 'lr', lr)

        if e % 1 == 0:
            lr = lr / 2

    plt.plot(loss_lst)
    plt.show()
    plt.plot(acc_lst)
    plt.show()

    img_data_test, lbl_data_test = get_test_data()
    img_data_test, mean, std = normalize(img_data_test, mean=mean, std=std)
    img_batches = np.split(img_data_test,
                        img_data_test.shape[0]//batch_size,
                        axis=0)
    label_batches = np.split(lbl_data_test,
                            lbl_data_test.shape[0]//batch_size,
                            axis=0)
    true_count = 0
    total_count = 0
    for no, img_batch in enumerate(img_batches):
        img_batch = np.reshape(img_batch, [img_batch.shape[0], -1])
        img_batch = np.expand_dims(img_batch, -1)
        labels = label_batches[no]

        for b in range(batch_size):
            x = img_batch
            # forward pass
            h1 = dense.forward(x)
            h1_nl = act1.forward(h1)
            h2 = dense2.forward(h1_nl)
            h2_nl = act2.forward(h2)
            h3 = dense3.forward(h2_nl)
            y_hat = act3.forward(h3)
            true = np.sum((labels == np.squeeze(np.argmax(y_hat, axis=1))
                        ).astype(np.float32))
            true_count += true
            total_count += batch_size
    print('test accuracy', true_count/total_count)
    print('done!')
