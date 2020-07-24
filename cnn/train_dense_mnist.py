import struct
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

    lr = 0.001
    batch_size = 50
    dense = DenseLayer(784, 1024)
    relu = ReLu()
    dense2 = DenseLayer(1024, 10)
    mse = MSELoss()
    iterations = 10
    loss_lst = []
    label_lst = []
    max_lst = []

    for i in range(iterations):

        for b in range(img_data_train.shape[0]):
            x = img_data_train[b].flatten()
            label = lbl_data_train[b]
            
            y = np.zeros([10])
            y[label] = 1

            h = dense.forward(x)
            h_nl = relu.forward(h)
            # h_nl = h
            y_hat = dense2.forward(h_nl)
            loss = mse.forward(y, y_hat)

            dl = mse.backward(y, y_hat)
            dw2, dx2 = dense2.backward(inputs=h_nl, prev_grad=dl)
            dx2 = relu.backward(dx2)
            dw, dx = dense.backward(inputs=x, prev_grad=dx2)


            label_lst.append(label)
            max_lst.append(np.argmax(y_hat))
            if b % batch_size == 0:
                dense.weight += -lr*dw
                dense2.weight += -lr*dw2
                dense.reset_grad()
                dense2.reset_grad()
                loss_lst.append(loss)

                true = np.sum(np.stack(label_lst) == np.stack(max_lst).astype(np.float32))
                acc = true/len(label_lst)
                print(i,b, loss, acc)
                max_lst = []
                label_lst = []

print(y)
print(y_hat)
plt.plot(loss_lst)
plt.show()

print('done')