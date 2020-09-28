# Created by moritz (wolter@cs.uni-bonn.de)
# a numpy only le net implementation
# based on http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
# and https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py

import struct
import numpy as np
import matplotlib.pyplot as plt
from im2col import im2col_indices, col2im_indices


def normalize(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data-mean)/std, mean, std


class CrossEntropyCost(object):

    def forward(self, label, out):
        return np.sum(np.nan_to_num(-label*np.log(out)-(1-label)*np.log(1-out)))

    def backward(self, label, out):
        return (out-label)


class MSELoss(object):
    ''' Mean squared error loss function. '''
    def forward(self, label, out):
        diff = out - label
        return np.mean(0.5*diff*diff)

    def backward(self, label, out):
        return out - label


class DenseLayer(object):
    def __init__(self, in_shape, out_shape):
        self.weight = np.zeros([1, out_shape, in_shape])
        self.weight = self.weight + np.random.randn(1, out_shape, in_shape)
        self.weight = self.weight / np.sqrt(in_shape)
        self.bias = np.random.randn(1, out_shape, 1)

    def forward(self, inputs):
        return np.matmul(self.weight, inputs) + self.bias

    def backward(self, inputs, prev_grad):
        dx = np.matmul(np.transpose(self.weight, [0, 2, 1]), prev_grad)
        dw = np.matmul(prev_grad, np.transpose(inputs, [0, 2, 1]))
        db = 1*prev_grad
        return dw, dx, db


class ReLu(object):

    def forward(self, inputs):
        inputs[inputs <= 0] = 0
        return inputs

    def backward(self, inputs, prev_dev):
        prev_dev[prev_dev <= 0] = 0
        return prev_dev


class Sigmoid(object):

    def sigmoid(self, inputs):
        return np.exp(inputs)/(1 + np.exp(inputs))

    def forward(self, inputs):
        return self.sigmoid(inputs)

    def backward(self, inputs, prev_dev):
        return self.sigmoid(inputs)*(1 - self.sigmoid(inputs))*prev_dev


class ConvLayer(object):

    def __init__(self, in_channels=None, out_channels=None,
                 height=None, width=None, stride=1, padding=0,
                 kernel=None, bias=None):
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = np.random.randn(out_channels, in_channels,
                                          height, width)
            self.kernel = self.kernel/np.sqrt(in_channels)
 
        self._stride = stride
        self._padding = padding
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.randn(1, out_channels, 1, 1)

    def convolution(self, img, kernel, stride):
        kernel_shape = kernel.shape
        kernel = kernel.flatten()
        kernel = np.expand_dims(kernel, -1)
        patches = skimage.util.view_as_windows(
            img, window_shape=kernel_shape, step=stride)
        patches_shape = patches.shape
        patches = np.reshape(patches,
                             [patches_shape[0]*patches_shape[1],
                              patches_shape[2]*patches_shape[3]])
        mul_conv = np.matmul(patches, kernel)
        return mul_conv

    def forward(self, img: np.array) -> np.array:
        """Compute a batched convolution forward pass.
        Args:
            img (np.array): The input 'image' [batch, channel, height, weight]
            kernel (np.array): The convolution kernel [out, in, height, width]
        Returns:
            np.array: The resulting output array with the convolution result.
        """
        kernel_shape = self.kernel.shape
        img_shape = img.shape

        out_height = int((img_shape[-2] + 2 * self._padding - kernel_shape[-2])
                         / self._stride + 1)
        out_width = int((img_shape[-1] + 2 * self._padding - kernel_shape[-1])
                        / self._stride + 1)
        # out_shape = [img_shape[0], kernel_shape[1], out_height, out_width]
        cols = im2col_indices(img, kernel_shape[-2], kernel_shape[-1],
                              padding=self._padding, stride=self._stride)
        kernel_flat = np.reshape(self.kernel,
                                 [kernel_shape[0], -1])
        cols = np.matmul(kernel_flat, cols)
        res = cols.reshape(kernel_shape[0], out_height,
                           out_width, img_shape[0])
        res = res.transpose(3, 0, 1, 2)
        res += self.bias
        return res

    def backward(self, inputs: np.array, prev_grad: np.array):
        """
        Args:
            img (np.array): The input 'image' [batch, channel, height, weight]
            prev_grad (np.array): The input gradients from the layer above.
        Returns:
            dx: (np.array): Gradient input into the next layer.
            dk: (np.array): Kernel gradient.
            db: (np.array): Bias gradient.
        """
        kernel_shape = self.kernel.shape
        img_shape = inputs.shape
        kernel_flat = np.reshape(self.kernel,
                                 [kernel_shape[0], -1])
        input_cols = im2col_indices(inputs,
                                    kernel_shape[-2], kernel_shape[-1],
                                    padding=self._padding, stride=self._stride)
        grad_cols = np.transpose(prev_grad, [1, 2, 3, 0])
        grad_cols = np.reshape(grad_cols, [kernel_shape[0], -1])
        dk = np.matmul(grad_cols, input_cols.T)
        dk = np.reshape(dk, self.kernel.shape)
        db = 1*prev_grad
        dx = np.matmul(kernel_flat.T, grad_cols)
        dx_shape = [img_shape[0], img_shape[1],
                    img_shape[-2], img_shape[-1]]
        dx = col2im_indices(dx, dx_shape,
                            kernel_shape[-2], kernel_shape[-1],
                            padding=self._padding, stride=self._stride)
        return dx, dk, db


def test_conv():
    img_data_train, lbl_data_train = get_train_data()
    kernel1 = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
    kernel2 = np.array([[1./9., 1./9., 1./9.],
                       [1./9., 1./9., 1./9.],
                       [1./9., 1./9., 1./9.]])
    bias = np.expand_dims(np.expand_dims(np.expand_dims(
        np.zeros([1]), -1), -1), -1)

    convlayer = ConvLayer(
        kernel=np.expand_dims(np.stack([kernel1, kernel2], 0), 1), stride=1,
        bias=bias)

    res = convlayer.forward(np.expand_dims(img_data_train[:50, :, :], 1))
    plt.imshow(res[0, 0, :, :])
    plt.show()
    plt.imshow(res[0, 1, :, :])
    plt.show()


def get_train_data():
    with open('cnn/data/t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        img_data_train = data.reshape((size, nrows, ncols))

    with open('cnn/data/t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.fromfile(f, dtype=np.dtype(np.uint8))
    return img_data_train, lbl_data_train


if __name__ == '__main__':
    img_data_train, lbl_data_train = get_train_data()
    img_data_train_norm, _, _ = normalize(img_data_train)
    idx = 55
    # plt.imshow(img_data_train[idx, :, :])
    # plt.title(str(lbl_data_train[idx]))
    # plt.show()

    test_conv()
    print('done')
