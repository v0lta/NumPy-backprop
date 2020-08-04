# a numpy only le net implementation
# based on http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
# and https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py

import struct
import numpy as np
import skimage.util
import matplotlib.pyplot as plt


def normalize(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data-mean)/std, mean, std


def im2col(img, kernel_shape, stride, padding='VALID'):
    assert padding == 'VALID'
    img_shape = img.shape
    patches = []
    for row_pos in range(0, img_shape[0]-kernel_shape[0]+1, stride):
        for col_pos in range(0, img_shape[1]-kernel_shape[1]+1, stride):
            row_strt = row_pos
            row_stop = int(row_pos + kernel_shape[0])
            col_strt = col_pos
            col_stop = int(col_pos + kernel_shape[1])
            img_patch = img[row_strt:row_stop, col_strt:col_stop]
            print(row_strt, row_stop, col_strt, col_stop)
            patches.append(img_patch.flatten())
    return np.stack(patches)


class CrossEntropyCost(object):

    def forward(self, label, out):
        return np.sum(np.nan_to_num(-label*np.log(out)-(1-label)*np.log(1-out)))

    def backward(self, label, out):
        return (out-label)


class MSELoss(object):
    '''
    The cross-entropy loss of the predictions
    '''

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
                 height=None, width=None, stride=1,
                 kernel=None, bias=None):
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = np.random.randn(out_channels, in_channels,
                                          height, width)
            self.kernel = self.kernel/np.sqrt(in_channels)
 
        self._stride = stride
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.randn(1, out_channels, 1, 1)/10.

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

    def batched_convolution(
            self, img: np.array, kernel: np.array,
            stride: int, pad: tuple = (0, 0),
            transpose=False) -> np.array:
        """Compute a batched convolution.
        Args:
            img (np.array): The input 'image' [batch, channel, height, weight]
            kernel (np.array): The convolution kernel [out, in, height, width]
            stride (int): The step size for the convolution.
            pad: (tuple): The number of zeros added to the left and right 
                              of the input sequence.
        Returns:
            np.array: The resulting output array with the convolution result.
        """
        kernel_shape = kernel.shape
        if pad != (0, 0,):
            img = np.pad(img, ((0, 0), (0, 0),
                               (pad[0], pad[0]),
                               (pad[1], pad[1])))

        window_shape = len(kernel_shape[:-2])*[1] + list(kernel_shape[-2:])
        patches = skimage.util.view_as_windows(img, window_shape=window_shape,
                                               step=stride)
        # test = skimage.util.view_as_windows(img[0],
        # window_shape=kernel_shape, step=stride)
        patches = np.squeeze(patches, axis=(4, 5))
        # create output dimension in patches.
        patches = np.expand_dims(patches, 1)
        patches_shape = patches.shape
        # reshape into:
        # [batch, out, channels, img_width*img_height, ker_width*ker_height]
        patches = np.reshape(patches,
                             list(patches_shape[:-4])
                             + [patches_shape[-4]*patches_shape[-3]]
                             + [patches_shape[-2]*patches_shape[-1]])

        kernel = np.reshape(kernel, [kernel_shape[0], kernel_shape[1], -1])
        if transpose:
            # [batch, out, channels,
            #  img_width*img_height, ker_width*ker_height]
            patches = np.transpose(patches, [0, 1, 2, 3, 4])
            kernel = np.expand_dims(np.expand_dims(kernel, -1), 2)
            mul_conv = np.matmul(patches, kernel)
            mul_conv = np.mean(mul_conv, axis=0)
        else:
            kernel = np.expand_dims(np.expand_dims(kernel, -1), 0)
            mul_conv = np.matmul(patches, kernel)
            mul_conv = np.squeeze(mul_conv, -1)
            # sum over the input channel dimension.
            mul_conv = np.sum(mul_conv, axis=2)
            mul_conv = np.reshape(mul_conv,
                                  [patches_shape[0], -1,
                                   patches_shape[3], patches_shape[4]])
        return mul_conv

    def forward(self, img):
        # batch_size = img.shape[0]
        # conv_lst = []
        # for b in range(batch_size):
        #     conv_lst.append(self.convolution(
        #         img[b], self.kernel, self._stride))
        # res_for = np.stack(conv_lst, axis=0)
        res = self.batched_convolution(img, self.kernel, self._stride)
        res += self.bias
        return res

    def backward(self, inputs, prev_grad):
        # todo replace matmul with conv?
        # dx = np.matmul(np.transpose(self.weight, [0, 2, 1]), prev_grad)
        # dw = np.matmul(prev_grad, np.transpose(inputs, [0, 2, 1]))
        t_pad_w = self.kernel.shape[-1] - 1
        t_pad_h = self.kernel.shape[-2] - 1
        flip_kernel = np.flip(np.flip(self.kernel, 2), 3)
        transpose_kernel = np.transpose(flip_kernel, [1, 0, 2, 3])
        dx = self.batched_convolution(
            prev_grad,
            kernel=transpose_kernel,
            stride=1,
            pad=(t_pad_w, t_pad_h))

        # dense_dx = np.matmul(np.transpose(np.squeeze(flip_kernel, -1)),
        #                      np.squeeze(prev_grad, -1))
        # print('test dx', np.sum(np.abs(dense_dx - np.squeeze(dx))))
        dk = self.batched_convolution(
             inputs,            # b, c, h, w
             kernel=prev_grad,  # out, in, h, w
             stride=self._stride,
             transpose=True)

        # dense_dk = np.matmul(np.squeeze(prev_grad, -1),
        #                      np.transpose(np.squeeze(inputs, -1), [0, 2, 1]))
        # print('test dk', np.sum(np.abs(np.squeeze(np.mean(dense_dk, 0))
        #                                - np.squeeze(dk))))
        db = 1*prev_grad
        return dx, dk, db


def test_patch():
    test_patches = im2col(img_data_train[0, :, :], kernel_shape=(3, 3),
                          stride=1)
    test_patches2 = skimage.util.view_as_windows(
        img_data_train[0, :, :], window_shape=(3, 3), step=(1, 1))
    patches_shape = test_patches2.shape
    test_patches2 = np.reshape(
        test_patches2,
        [patches_shape[0]*patches_shape[1], patches_shape[2]*patches_shape[3]])
    print('err', np.sum(np.abs(test_patches - test_patches2)))


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

    test_patch()
    test_conv()
    print('done')
