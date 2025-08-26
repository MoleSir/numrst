use crate::{FloatDType, NdArray, Result};

pub fn lu<T: FloatDType>(arr: &NdArray<T>) -> Result<(NdArray<T>, NdArray<T>)> {
    let (n, _) = arr.dims2()?;
    let l = crate::linalg::eye::<T>(n)?;
    let u = arr.zero_like()?;

    let arrm = arr.matrix_view()?;
    let lm = l.matrix_view()?;
    let um = u.matrix_view()?;

    for i in 0..n {
        // U
        for j in i..n {
            let sum = lm.row(i)?.dot(&um.col(j)?)?;
            let arr_v = arrm.get(i, j)?; 
            um.set(i, j, arr_v - sum)?;
        }

        // L
        for j in i+1..n {
            let sum = lm.row(j)?.dot(&um.col(i)?)?;
            let arr_v = arrm.get(j, i)?;
            let u_v = um.get(i, i)?;
            lm.set(j, i, (arr_v - sum) / u_v)?;
        }
    }

    Ok((l, u))
}

#[cfg(test)]
mod test {
    use crate::NdArray;

    #[test]
    fn test_lu() {
        let arr = NdArray::new(&[
            [2., 3., 1.],
            [4., 7., 3.],
            [6., 18., 5.],
        ]).unwrap();
        let (l, u) = crate::linalg::lu(&arr).unwrap();
        let arr_rec = l.matmul(&u).unwrap();
        println!("{}", arr_rec);
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }
}