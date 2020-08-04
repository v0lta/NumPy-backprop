import numpy as np
import matplotlib.pyplot as plt
from numpy_layer import ConvLayer
from numpy_layer import MSELoss
from numpy_layer import ReLu
from numpy_layer import Sigmoid


x = np.linspace(0, 10, num=100)
y = np.sin(x) + np.random.uniform(-0.1, 0.1, size=(100))
x = np.expand_dims(np.expand_dims(x, -1), 0)
y = np.expand_dims(np.expand_dims(y, -1), 0)
lr = 0.0001

# plt.plot(x, y)
plt.show()

layer = ConvLayer(in_channels=1, out_channels=150,
                  height=1, width=1, stride=1)
act = Sigmoid()
layer2 = ConvLayer(in_channels=150, out_channels=1,
                   height=1, width=1, stride=1)
mse = MSELoss()
batch_size = 5
iterations = 100000//batch_size


# plt.plot(np.squeeze(x), np.squeeze(y))
# plt.plot(np.squeeze(x), np.squeeze(layer.forward(x)))
# plt.title('init')
# plt.show()


for i in range(iterations):
    x_lst = []
    y_lst = []
    for b in range(batch_size):
        x = np.linspace(0, 10, num=100)
        y = np.sin(x) + np.random.uniform(-0.1, 0.1, size=(100))
        x_lst.append(x)
        y_lst.append(y)
    x = np.expand_dims(np.expand_dims(np.stack(x_lst, axis=0), axis=-1),
                       axis=1)
    y = np.expand_dims(np.expand_dims(np.stack(y_lst, axis=0), axis=-1),
                       axis=1)

    # forward
    h = layer.forward(x)
    h_nl = act.forward(h)
    y_hat = layer2.forward(h_nl)
    loss = mse.forward(y, y_hat)

    # backward
    dl = mse.backward(y, y_hat)
    dx2, dk2, db2 = layer2.backward(inputs=h_nl, prev_grad=dl)
    dx2 = act.backward(inputs=h, prev_dev=dx2)
    dx, dk, db = layer.backward(inputs=x, prev_grad=dx2)

    # update
    layer.kernel += -lr*dk
    layer.bias += -lr*np.expand_dims(np.mean(db, axis=(0, 2)), (0, 2))
    layer2.kernel += -lr*dk2
    layer2.bias += -lr*np.expand_dims(np.mean(db2, axis=(0, 2)), (0, 2))
    if i % 10 == 0:
        print(i, loss)


x = np.linspace(0, 10, num=100)
y = np.sin(x) + np.random.uniform(-0.1, 0.1, size=(100))
# x = x[2:-2]
# y = y[2:-2]
x = np.expand_dims(x, (0, 1, 3))
y = np.expand_dims(y, (0, 1, 3))

plt.plot(np.squeeze(x), np.squeeze(y))
plt.plot(np.squeeze(x),
         np.squeeze(layer2.forward(act.forward(layer.forward(x)))))
plt.title('opt')
plt.show()

print('done')
