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
    with open('cnn/data/t10k-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        img_data_test = data.reshape((size, nrows, ncols))

    with open('cnn/data/t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        lbl_data_test = np.fromfile(f, dtype=np.dtype(np.uint8))
    return img_data_test, lbl_data_test


def get_train_data():
    with open('cnn/data/train-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        img_data_train = data.reshape((size, nrows, ncols))

    with open('cnn/data/train-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.fromfile(f, dtype=np.dtype(np.uint8))
    return img_data_train, lbl_data_train




if __name__ == '__main__':
    img_data_train, lbl_data_train = get_train_data()
    img_data_train, mean, std = normalize(img_data_train)

    lr = 1.0
    batch_size = 100
    dense = DenseLayer(784, 256)
    act1 = ReLu()
    dense2 = DenseLayer(256, 128)
    act2 = ReLu()
    dense3 = DenseLayer(128, 10)
    act3 = Sigmoid()
    cost = CrossEntropyCost()
    iterations = 10
    loss_lst = []
    acc_lst = []

    for e in range(iterations):
        shuffler = np.random.permutation(len(img_data_train))
        img_data_train = img_data_train[shuffler]
        lbl_data_train = lbl_data_train[shuffler]

        img_batches = np.split(img_data_train, img_data_train.shape[0]//batch_size, axis=0)
        label_batches = np.split(lbl_data_train, lbl_data_train.shape[0]//batch_size, axis=0)

        #for b in range(img_data_train.shape[0]):
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
            h1 = dense.forward(x)
            h1_nl = act1.forward(h1)
            h2 = dense2.forward(h1_nl)
            h2_nl = act2.forward(h2)
            h3 = dense3.forward(h2_nl)
            y_hat = act3.forward(h3)
            loss = cost.forward(label=y, out=y_hat)

            dl = cost.backward(label=y, out=y_hat)
            # dl = act3.backward(inputs=h3, prev_dev=dl)
            dw3, dx3 = dense3.backward(inputs=h2_nl, prev_grad=dl)
            dx3 = act2.backward(dx3)
            dw2, dx2 = dense2.backward(inputs=h1_nl, prev_grad=dx3)
            dx2 = act1.backward(dx2)
            dw, _ = dense.backward(inputs=x, prev_grad=dx2)

            dense.weight += -lr*np.mean(dw, axis=0)
            dense2.weight += -lr*np.mean(dw2, axis=0)
            dense3.weight += -lr*np.mean(dw3, axis=0)
            loss_lst.append(loss)

            true = np.sum((labels == np.squeeze(np.argmax(y_hat, axis=1))).astype(np.float32))
            acc = true/batch_size
            acc_lst.append(acc)
            if no % 5 == 0:
                print('e', e,'b', no, 'loss', loss,'acc', acc, 'lr', lr)

        if e % 1 == 0 and e > 0: 
            lr = lr / 2

plt.plot(loss_lst)
plt.show()
plt.plot(acc_lst)
plt.show()

print('done')