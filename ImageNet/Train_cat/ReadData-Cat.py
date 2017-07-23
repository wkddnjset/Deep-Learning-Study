import numpy as np
import tensorflow as tf

cat_x = np.fromfile('cat_all.dat', dtype='uint8')
cat_y = np.fromfile('cat_label_all.dat', dtype='float32')
cat_x = np.reshape(cat_x, [-1, 400, 400, 3])
cat_y = np.reshape(cat_y, [-1, 2])
print(cat_x.shape)
print(cat_x.dtype)
print(cat_y.shape)
print(cat_y.dtype)