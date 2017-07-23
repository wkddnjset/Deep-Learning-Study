import numpy as np

label = []
for i in range(0, 872):
    label.append([1, 0])
    label_cat = np.array(label)
    fin = label_cat.astype('float32')

print(fin.dtype)

fin.tofile('cat_label_all.dat')