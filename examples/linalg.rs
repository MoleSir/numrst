use numrst::{linalg, NdArray};

fn validate_bidiag(a: &NdArray<f64>, b: &NdArray<f64>, u: &NdArray<f64>, v: &NdArray<f64>, upper: bool) {
    let bv = b.matrix_view().unwrap();
    let (m, n) = bv.shape();
    for i in 0..m {
        for j in 0..n {
            let allowed = if upper {
                j == i || j == i + 1
            } else {
                j == i || j + 1 == i
            };
            if !allowed {
                assert!(bv.g(i, j).abs() < 1e-10, "B should be bidiagonal, but got {} at ({}, {})", bv.g(i, j), i, j);
            }
        }
    }

    let recovered = u.matmul(b).unwrap().matmul(&v.transpose_last().unwrap()).unwrap();
    assert!(recovered.allclose(a, 1e-5, 1e-5), "A and U*B*V^T mismatch");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = NdArray::new(&[
        [1., 2., 3., 4., 5.],
        [2., 4., 1., 2., 1.],
        [3., 1., 7., 1., 1.],
        [4., 2., 1., -1., 3.],
        [5., 1., 1., 3., 2.],
    ]).unwrap();

    let (b, u, v) = linalg::bidiagonal(&a).unwrap();
    validate_bidiag(&a, &b, &u, &v, true);

    Ok(())
}
