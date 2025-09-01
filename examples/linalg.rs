use numrst::NdArray;
use numrst::linalg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arr = NdArray::new(&[
        [2., 3., 1.],
        [4., 7., 3.],
    ]).unwrap(); // 2x3

    let (l, u) = linalg::lu(&arr).unwrap();
    let l = l.to_ndarray();
    let u = u.to_ndarray();

    let arr_rec = l.matmul(&u).unwrap();
    assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));

    Ok(())
}
