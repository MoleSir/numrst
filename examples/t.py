import numpy as np
from scipy.linalg import lu

A = np.array([[0, 1],
              [1, 0]], dtype=float)

P, L, U = lu(A)

print("P =\n", P)
print("L =\n", L)
print("U =\n", U)
