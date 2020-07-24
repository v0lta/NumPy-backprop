import struct
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy_layer import normalize
from numpy_layer import DenseLayer
from numpy_layer import MSELoss
from numpy_layer import ReLu


def get_train_data():
    with open('cnn/data/t10k-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        img_data_train = data.reshape((size, nrows, ncols))

    with open('cnn/data/t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.fromfile(f, dtype=np.dtype(np.uint8))
    return img_data_train, lbl_data_train


if __name__ == '__main__':
    img_data_train, lbl_data_train = get_train_data()
    img_data_train, mean, var = normalize(img_data_train)

    lr = 0.01
    batch_size = 250
    dense = DenseLayer(784, 1024)
    relu = ReLu()
    dense2 = DenseLayer(1024, 10)
    mse = MSELoss()
    iterations = 30
    loss_lst = []
    acc_lst = []

    for i in range(iterations):
        img_lst = np.split(img_data_train, img_data_train.shape[0], axis=0)
        lbl_lst = np.split(lbl_data_train, lbl_data_train.shape[0], axis=0)
        merge_lst = list(zip(img_lst, lbl_lst))
        random.shuffle(merge_lst)
        img_lst, lbl_lst = zip(*merge_lst)
        img_data_train = np.stack(img_lst)
        lbl_data_train = np.stack(lbl_data_train)

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
            h = dense.forward(x)
            h_nl = relu.forward(h)
            # h_nl = h
            y_hat = dense2.forward(h_nl)
            loss = mse.forward(y, y_hat)

            dl = mse.backward(y, y_hat)
            dw2, dx2 = dense2.backward(inputs=h_nl, prev_grad=dl)
            dx2 = relu.backward(dx2)
            dw, dx = dense.backward(inputs=x, prev_grad=dx2)


            dense.weight += -lr*np.mean(dw, axis=0)
            dense2.weight += -lr*np.mean(dw2, axis=0)
            loss_lst.append(loss)

            true = np.sum((labels == np.squeeze(np.argmax(y_hat, axis=1))).astype(np.float32))
            acc = true/batch_size
            acc_lst.append(acc)
            print(i,no, loss, acc)


plt.plot(loss_lst)
plt.show()

print('done')