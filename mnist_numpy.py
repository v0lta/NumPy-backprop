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

# Set up a lenet

