# Created by moritz (wolter@cs.uni-bonn.de)
import numpy as np
import matplotlib.pyplot as plt
from numpy_layer import DenseLayer
from numpy_layer import MSELoss
from numpy_layer import ReLu
from numpy_layer import Sigmoid


x = np.linspace(0, 10, num=100)
y = np.sin(x) + np.random.uniform(-0.1, 0.1, size=(100))
x = np.expand_dims(np.expand_dims(x, -1), 0)
y = np.expand_dims(np.expand_dims(y, -1), 0)
lr = 0.001

# plt.plot(x, y)
plt.show()

dense = DenseLayer(100, 150)
act = Sigmoid()
dense2 = DenseLayer(150, 100)
mse = MSELoss()
batch_size = 5
iterations = 10000//batch_size

plt.plot(np.squeeze(x), np.squeeze(y))
plt.plot(np.squeeze(x),
         np.squeeze(dense2.forward(act.forward(dense.forward(x)))))
plt.title('init')
plt.show()


for i in range(iterations):
    x_lst = []
    y_lst = []
    for b in range(batch_size):
        x = np.linspace(0, 10, num=100)
        y = np.sin(x) + np.random.uniform(-0.1, 0.1, size=(100))
        x_lst.append(x)
        y_lst.append(y)
    x = np.expand_dims(np.stack(x_lst, axis=0), axis=-1)
    y = np.expand_dims(np.stack(y_lst, axis=0), axis=-1)

    # forward
    h = dense.forward(x)
    h_nl = act.forward(h)
    # h_nl = h
    y_hat = dense2.forward(h_nl)
    loss = mse.forward(y, y_hat)

    # backward
    dl = mse.backward(y, y_hat)
    grad_dense2 = dense2.backward(inputs=h_nl, delta=dl)
    dx2 = act.backward(inputs=h, delta=grad_dense2['x'])
    grad_dense = dense.backward(inputs=x, delta=dx2)

    # update
    dense.weights['W'] += -lr*np.mean(grad_dense['W'], axis=0)
    dense.weights['b'] += -lr*np.mean(grad_dense['b'], axis=0)
    dense2.weights['W'] += -lr*np.mean(grad_dense2['W'], axis=0)
    dense2.weights['b'] += -lr*np.mean(grad_dense2['b'], axis=0)
    if i % 100 == 0:
        print(i, loss)


x = np.linspace(0, 10, num=100)
y = np.sin(x) + np.random.uniform(-0.1, 0.1, size=(100))
x = np.expand_dims(np.expand_dims(x, -1), 0)
y = np.expand_dims(np.expand_dims(y, -1), 0)
plt.plot(np.squeeze(x), np.squeeze(y))
plt.plot(np.squeeze(x), np.squeeze(dense2.forward(act.forward(dense.forward(x)))))
plt.title('opt')
plt.show()

print('done')