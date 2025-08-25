use num_traits::Zero;
use crate::{Error, Layout, Result, Shape, Storage, WithDType};
use super::NdArray;

impl<T: WithDType> NdArray<T> {
    /// Returns the matrix-multiplication of the input tensor with the other provided tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A tensor with dimensions `b1, b2, ..., bi, m, k`.
    /// * `rhs` - A tensor with dimensions `b1, b2, ..., bi, k, n`.
    ///
    /// The resulting tensor has dimensions `b1, b2, ..., bi, m, n`.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let a_dims = self.shape().dims();
        let b_dims = rhs.shape().dims();

        let dim = a_dims.len();

        if dim < 2 || b_dims.len() != dim {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            })?
        }

        let m = a_dims[dim - 2];
        let k = a_dims[dim - 1];
        let k2 = b_dims[dim - 2];
        let n = b_dims[dim - 1];

        let c_shape = Shape::from(&a_dims[..dim - 2]).extend(&[m, n]);
        if c_shape.element_count() == 0 || k == 0 {
            return Self::zeros(c_shape);
        }
        let batching: usize = a_dims[..dim - 2].iter().product();
        let batching_b: usize = b_dims[..dim - 2].iter().product();
        if k != k2 || batching != batching_b {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            })?
        }

        // (..., m, k) @ (..., k, n)
        let c_storage = Self::do_matmul(
            &self.storage(),
            self.layout(),
            &rhs.storage(),
            rhs.layout(),
            (batching, m, n, k),
        );

        Ok(Self::from_storage(c_storage, c_shape))
    }

    fn do_matmul(
        lhs: &Storage<T>, lhs_layout: &Layout, 
        rhs: &Storage<T>, rhs_layout: &Layout, 
        bmnk: (usize, usize, usize, usize)
    ) -> Storage<T> 
        where T: num_traits::Num + Copy + Zero
    {
        let lhs = lhs.data();
        let rhs = rhs.data();

        // l (b.., m, k)
        // r (b.., k, n)
        let lhs_rank = lhs_layout.shape().rank();
        let rhs_rank = rhs_layout.shape().rank();
        assert!(lhs_rank == rhs_rank && rhs_rank >= 2);
        let (bs, ms, ns, ks) = bmnk;
        assert_eq!(lhs_layout.dims()[lhs_rank - 2], ms);
        assert_eq!(lhs_layout.dims()[lhs_rank - 1], ks);
        assert_eq!(rhs_layout.dims()[rhs_rank - 2], ks);
        assert_eq!(rhs_layout.dims()[rhs_rank - 1], ns);
        let mut dst = vec![T::zero(); bs * ms * ns];

        let l_batch_stride = ms * lhs_layout.stride()[lhs_rank - 2];
        let r_batch_stride = ks * rhs_layout.stride()[rhs_rank - 2];
        let dst_batch_stride = ms * ns;

        let l_last_stride = lhs_layout.stride()[lhs_rank - 1];
        let r_last_stride = rhs_layout.stride()[rhs_rank - 1];
        let l_sec_last_stride = lhs_layout.stride()[lhs_rank - 2];
        let r_sec_last_stride = rhs_layout.stride()[rhs_rank - 2];

        // For each batch
        for b in 0..bs {
            let lhs_ = &lhs[lhs_layout.start_offset() + b * l_batch_stride .. ];
            let rhs_ = &rhs[rhs_layout.start_offset() + b * r_batch_stride .. ]; 
            let dst_ = &mut dst[b * dst_batch_stride .. ];

            for m in 0..ms {
                // lhs_'s m row
                let row = &lhs_[m * l_sec_last_stride ..];

                for n in 0..ns {
                    // rhs_'s n col
                    let col = &rhs_[n * r_last_stride ..];

                    let mut v = T::zero();
                    for k in 0..ks {
                        // row's k  * col's k
                        let l = row[k * l_last_stride];
                        let r = col[k * r_sec_last_stride];
                        v = v + l * r;
                    }
                    dst_[m * ns + n] = v;
                }
            }
        }
        
        Storage::new(dst)
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use crate::{rng, DType, IndexOp, Range};

    use super::*;

    #[test]
    fn test_matmul_2d() {
        // A: (2, 3), B: (3, 2)
        let a = NdArray::arange(0, 6).unwrap().reshape((2, 3)).unwrap(); // [[0,1,2],[3,4,5]]
        let b = NdArray::arange(0, 6).unwrap().reshape((3, 2)).unwrap(); // [[0,1],[2,3],[4,5]]
        let c = a.matmul(&b).unwrap();

        let expected = NdArray::new(&[
            [0*0 + 1*2 + 2*4, 0*1 + 1*3 + 2*5], // [10, 13]
            [3*0 + 4*2 + 5*4, 3*1 + 4*3 + 5*5], // [28, 40]
        ]).unwrap();

        assert!(c.allclose(&expected, 1e-5, 1e-8));
    }


    #[test]
    fn test_matmul_batch() {
        // A: (2, 2, 3), B: (2, 3, 2)
        let a = NdArray::arange(0., 12.).unwrap().reshape((2, 2, 3)).unwrap();
        let b = NdArray::arange(0., 12.).unwrap().reshape((2, 3, 2)).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.dims(), &[2, 2, 2]);

        // batch 0
        let a0 = NdArray::new(&[[0.,1.,2.],[3.,4.,5.]]).unwrap();
        let b0 = NdArray::new(&[[0.,1.],[2.,3.],[4.,5.]]).unwrap();
        let c0 = a0.matmul(&b0).unwrap();

        // batch 1
        let a1 = NdArray::new(&[[6.,7.,8.],[9.,10.,11.]]).unwrap();
        let b1 = NdArray::new(&[[6.,7.],[8.,9.],[10.,11.]]).unwrap();
        let c1 = a1.matmul(&b1).unwrap();

        assert!(c0.allclose(&c.index(0).unwrap(), 1e-5, 1e-8));
        assert!(c1.allclose(&c.index(1).unwrap(), 1e-5, 1e-8));
    }

    #[test]
    fn test_matmul_not_continues() {
        let a = NdArray::arange(0., 125.).unwrap().reshape((5, 5, 5)).unwrap();

        let sub_a = a.index((rng!(1:3), rng!(3:5), 2)).unwrap();
        let mut vals = Vec::new();
        for i in 1..3 {
            for j in 3..5 {
                vals.push((i * 25 + j * 5 + 2) as f64);
            }
        }
        let expected = NdArray::from_vec(vals, (2, 2)).unwrap();
        assert!(sub_a.allclose(&expected, 0.0, 0.0));

        let b = NdArray::randn(0.0, 1.0, (2, 5)).unwrap();

        let res = sub_a.matmul(&b).unwrap();
        let res_expected = expected.matmul(&b).unwrap();
        assert!(res.allclose(&res_expected, 1e-5, 1e-8));
    }
}