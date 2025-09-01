import numpy as np
import scipy

A = np.array([
    [2., 4., 1.],
    [4., 8., 2.],
    [1., 2., 0.5]
])

P, L, U = scipy.linalg.lu(A)   # 注意 NumPy 里通常用 scipy.linalg.lu
print(P)
print(L)
print(U)