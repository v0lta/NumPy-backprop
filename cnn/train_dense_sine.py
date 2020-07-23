import numpy as np
import matplotlib.pyplot as plt
from numpy_layer import DenseLayer
from numpy_layer import MSELoss
from numpy_layer import ReLu


x = np.linspace(0, 10, num=100)
y = np.sin(x) + np.random.uniform(-0.1, 0.1, size=(100))
lr = 0.01

# plt.plot(x, y)
plt.show()

dense = DenseLayer(100, 100)
relu = ReLu()
dense2 = DenseLayer(100, 100)
mse = MSELoss()
iterations = 8000


plt.plot(x, y)
plt.plot(x, dense.forward(x))
plt.title('init')
plt.show()


for i in range(iterations):

    x = np.linspace(0, 10, num=100)
    y = np.sin(x) + np.random.uniform(-0.1, 0.1, size=(100))

    h = dense.forward(x)
    h_nl = relu.forward(h)
    # h_nl = h
    y_hat = dense2.forward(h_nl)
    loss = mse.forward(y, y_hat)

    dl = mse.backward(y, y_hat)
    dw2, dx2 = dense2.backward(inputs=h_nl, prev_grad=dl)
    dx2 = relu.backward(dx2)
    dw, dx = dense.backward(inputs=x, prev_grad=dx2)
    dense.weight += -lr*dw
    dense2.weight += -lr*dw2
    if i % 100 == 0:
        print(i, loss)


x = np.linspace(0, 10, num=100)
y = np.sin(x) + np.random.uniform(-0.1, 0.1, size=(100))
plt.plot(x, y)
plt.plot(x, dense2.forward(relu.forward(dense.forward(x))))
plt.title('opt')
plt.show()
