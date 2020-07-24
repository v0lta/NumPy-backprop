# a numpy only le net implementation
# based on http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

import struct
import numpy as np
import skimage.util
import matplotlib.pyplot as plt


def normalize(data):
    mean = np.mean(data)
    var = np.mean(data)
    return (data-mean)/var, mean, var


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


class MSELoss(object):
    '''
    The cross-entropy loss of the predictions
    '''
    def init():
        super()

    def forward(self, label, out):
        diff = out - label
        return np.mean(0.5*diff*diff)

    def backward(self, label, out):
        return out - label


class DenseLayer(object):
    def __init__(self, in_shape, out_shape):
        self.weight = np.zeros([1, out_shape, in_shape])
        self.weight += np.random.uniform(-0.001, 0.001, [1, out_shape, in_shape])
        
    def forward(self, inputs):
        return np.matmul(self.weight, inputs)

    def backward(self, inputs, prev_grad):
        dw = np.matmul(prev_grad, np.transpose(inputs, [0, 2, 1]))
        dx = np.matmul(np.transpose(self.weight, [0, 2, 1]), prev_grad)
        return dw, dx


class ReLu(object):

    def forward(self, inputs):
        inputs[inputs <= 0] = 0
        return inputs

    def backward(self, prev_dev):
        prev_dev[prev_dev <= 0] = 0
        return prev_dev


class ConvLayer(object):

    def __init__(self, kernel, stride):
        self._kernel = kernel
        self._stride = stride

    def convolution(self, img, kernel, stride):
        kernel_shape = kernel.shape
        kernel = kernel.flatten()
        kernel = np.expand_dims(kernel, -1)
        patches = skimage.util.view_as_windows(
            img_data_train[0, :, :], window_shape=kernel_shape, step=stride)
        patches_shape = patches.shape
        patches = np.reshape(patches, 
                             [patches_shape[0]*patches_shape[1],
                              patches_shape[2]*patches_shape[3]])
        mul_conv = np.matmul(patches, kernel)
        return mul_conv

    def forward(self, img):
        batch_size = img.shape[0]
        conv_lst = []
        for b in range(batch_size):
            conv_lst.append(self.convolution(
                img[b], self._kernel, self._stride))
        return np.stack(conv_lst, axis=0)

    def backward(self, img, dev):
        pass


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
    kernel = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]])
    conv = ConvLayer.convolution(kernel=kernel, img=img_data_train[0, :, :],
                                 stride=1)
    plt.imshow(conv.reshape(26, 26))
    plt.title('conv_result '+str(lbl_data_train[0]))
    plt.show()

    kernel = np.array([[1./9., 1./9., 1./9.],
                      [1./9., 1./9., 1./9.],
                      [1./9., 1./9., 1./9.]])
    conv = ConvLayer.convolution(kernel=kernel, img=img_data_train[0, :, :],
                                 stride=1)
    plt.imshow(conv.reshape(26, 26))
    plt.title('mean_conv_result '+str(lbl_data_train[0]))
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
    img_data_train_norm = normalize(img_data_train)
    idx = 55
    print(lbl_data_train[idx])
    plt.imshow(img_data_train_norm[idx, :, :])
    plt.show()

    # print(lbl_data_train[idx][0])
    plt.imshow(img_data_train[idx, :, :])
    plt.title(str(lbl_data_train[idx]))
    plt.show()

    test_patch()
    # test_conv()
    print('done')
