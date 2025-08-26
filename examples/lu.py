import numpy as np

def lu_decomposition(A):
    A = A.copy()
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros_like(A)

    for i in range(n):
        # U 上三角
        for j in range(i, n):
            U[i,j] = A[i,j] - sum(L[i,k]*U[k,j] for k in range(i))
        # L 下三角
        for j in range(i+1, n):
            L[j,i] = (A[j,i] - sum(L[j,k]*U[k,i] for k in range(i))) / U[i,i]

    return L, U

A = np.array([[2, 3, 1],
              [4, 7, 3],
              [6, 18, 5]], dtype=float)

L, U = lu_decomposition(A)
print("L:\n", L)
print("U:\n", U)

# 验证
print(np.allclose(A, L @ U))

