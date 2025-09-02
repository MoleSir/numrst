import numpy as np

def test_svd_rank_deficient():
    # rank 2 < 3
    a = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 1.0, 1.0]
    ])
    
    # 执行 SVD
    U, s, Vt = np.linalg.svd(a, full_matrices=False)
    V = Vt.T

    print(U)
    print(s)
    print(V)


    # 1. 奇异值应有一个接近 0
    zero_sigma_count = np.sum(s < 1e-5)
    assert zero_sigma_count >= 1, f"Expected at least 1 near-zero singular value, got {s}"

    # 2. 重构矩阵近似原矩阵
    rec = U @ np.diag(s) @ V.T
    assert np.allclose(rec, a, rtol=1e-7, atol=1e-7), f"Reconstruction failed\n{rec}\n{a}"

    # 3. U 和 V 正交
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), rtol=1e-6, atol=1e-6), "U is not orthogonal"
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), rtol=1e-6, atol=1e-6), "V is not orthogonal"

    print(U.T @ U)

    print("test_svd_rank_deficient passed.")


# 执行测试
if __name__ == "__main__":
    test_svd_rank_deficient()
