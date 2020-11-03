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
        return -np.mean(np.nan_to_num(
            label*np.log(out) + (1-label)*np.log(1-out)))

    def backward(self, label, out):
        """ Assuming a sigmoidal netwok output."""
        return (out-label)


class MSELoss(object):
    ''' Mean squared error loss function. '''
    def forward(self, label, out):
        diff = out - label
        return np.mean(diff*diff)

    def backward(self, label, out):
        return out - label


class DenseLayer(object):
    def __init__(self, in_shape, out_shape):
        self.weights = {}
        W = np.zeros([1, out_shape, in_shape])
        W = W + np.random.randn(1, out_shape, in_shape)
        W = W / np.sqrt(in_shape)
        self.weights['W'] = W
        b = np.random.randn(1, out_shape, 1)
        self.weights['b'] = b


    def forward(self, inputs):
        return np.matmul(self.weights['W'], inputs) + self.weights['b']

    def backward(self, inputs, delta) -> {}:
        """Backward pass through a dense layer.
        Args:
            inputs: [batch_size, input_dim, 1]
            delta: [batch_size, out_dim, 1]
        Returns:
            A dictionary containing:
            'W' - dW (np.array): Weight gradients
            'b' - db (np.array): bias gradients
            'x' - dx (np.array): input gradients for lower layers.
        """
        dx = np.matmul(np.transpose(self.weights['W'], [0, 2, 1]), delta)
        dw = np.matmul(delta, np.transpose(inputs, [0, 2, 1]))
        db = 1*delta
        return {'W': dw, 'b': db, 'x': dx}


class ReLu(object):

    def forward(self, inputs):
        inputs[inputs <= 0] = 0
        return inputs

    def backward(self, inputs, prev_dev):
        prev_dev[prev_dev <= 0] = 0
        return prev_dev

class Sigmoid(object):
    """ Sigmoid activation function. """
    def sigmoid(self, inputs):
        # sig = np.exp(inputs)/(1 + np.exp(inputs))
        # return np.nan_to_num(sig)
        return np.where(inputs >= 0, 
                        1 / (1 + np.exp(-inputs)), 
                        np.exp(inputs) / (1 + np.exp(inputs)))

    def forward(self, inputs):
        return self.sigmoid(inputs)

    def backward(self, inputs, delta):
        return self.sigmoid(inputs)*(1 - self.sigmoid(inputs))*delta

    def prime(self, inputs):
        return self.sigmoid(inputs)*(1 - self.sigmoid(inputs))

class ConvLayer(object):

    def __init__(self, in_channels=None, out_channels=None,
                 height=None, width=None, stride=1, padding=0):
        self._stride = stride
        self._padding = padding
        self.weights = {}
        kernel = np.random.randn(out_channels, in_channels,
                                      height, width)
        kernel = kernel/np.sqrt(in_channels)
        self.weights['K'] = kernel
        bias = np.random.randn(1, out_channels, 1, 1)
        self.weights['b'] = bias

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
        kernel_shape = self.weights['K'].shape
        img_shape = img.shape

        out_height = int((img_shape[-2] + 2 * self._padding - kernel_shape[-2])
                         / self._stride + 1)
        out_width = int((img_shape[-1] + 2 * self._padding - kernel_shape[-1])
                        / self._stride + 1)
        cols = im2col_indices(img, kernel_shape[-2], kernel_shape[-1],
                              padding=self._padding, stride=self._stride)
        kernel_flat = np.reshape(self.weights['K'],
                                 [kernel_shape[0], -1])
        cols = np.matmul(kernel_flat, cols)
        res = cols.reshape(kernel_shape[0], out_height,
                           out_width, img_shape[0])
        res = res.transpose(3, 0, 1, 2)
        res += self.weights['b']
        return res

    def backward(self, inputs: np.array, delta: np.array) -> {}:
        """
        Args:
            img (np.array): The input 'image' [batch, channel, height, weight]
            delta (np.array): The input gradients from the layer above.
        Returns:
            A dictionary containing:
            'x' - dx: (np.array): Gradient input into the next layer.
            'K' - dk: (np.array): Kernel gradient.
            'b' - db: (np.array): Bias gradient.
        """
        kernel_shape = self.weights['K'].shape
        img_shape = inputs.shape
        kernel_flat = np.reshape(self.weights['K'],
                                 [kernel_shape[0], -1])
        input_cols = im2col_indices(inputs,
                                    kernel_shape[-2], kernel_shape[-1],
                                    padding=self._padding, stride=self._stride)
        grad_cols = np.transpose(delta, [1, 2, 3, 0])
        grad_cols = np.reshape(grad_cols, [kernel_shape[0], -1])
        dk = np.matmul(grad_cols, input_cols.T)
        dk = np.reshape(dk, self.weights['K'].shape)
        db = 1*delta
        dx = np.matmul(kernel_flat.T, grad_cols)
        dx_shape = [img_shape[0], img_shape[1],
                    img_shape[-2], img_shape[-1]]
        dx = col2im_indices(dx, dx_shape,
                            kernel_shape[-2], kernel_shape[-1],
                            padding=self._padding, stride=self._stride)
        return {'x': dx,'K': dk,'b': db}






























