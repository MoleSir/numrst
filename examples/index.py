import numpy as np

a = np.arange(5*5*5).reshape(5,5,5)
# a.shape -> (5, 5, 5)

idx = [0, 1, 4]
b = a[idx, :, :]

b.shape  # (5, 3, 5)

print(a)
print(b)

print(a.strides)
print(b.strides)

