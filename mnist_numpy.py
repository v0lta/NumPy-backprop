# a numpy only le net implementation
# based on http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

import ipdb
import struct
import numpy as np
import matplotlib.pyplot as plt

with open('data/t10k-images-idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    img_data_train = data.reshape((size, nrows, ncols))

with open('data/t10k-labels-idx1-ubyte', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    lbl_data_train = np.fromfile(f, dtype=np.dtype(np.uint8))

idx = 55
print(lbl_data_train[idx])
plt.imshow(img_data_train[idx, :, :])
plt.show()


def normalize(data):
    mean = np.mean(data)
    var = np.mean(data)
    return (data-mean)/var


img_data_train_norm = normalize(img_data_train)

print(lbl_data_train[idx])
plt.imshow(img_data_train_norm[idx, :, :])
plt.show()


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


def col2im(col_res, img_shape, kernel_shape, stride, padding='VALID'):
    assert padding == 'VALID'
    rows = img_shape[0] 
    cols = None
    return None



def convolution(kernel, img):
    # im2col
    # matmul

    pass


test_patches = im2col(img_data_train[0, :, :], kernel_shape=(3, 3),
                      stride=1)
kernel = np.random.randn(3, 3)
kernel = kernel.flatten()
kernel = np.expand_dims(kernel, -1)
mul_conv = np.matmul(test_patches, kernel)
