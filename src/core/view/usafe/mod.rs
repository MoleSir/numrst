mod matrix;
mod vector;
pub use matrix::*;
pub use vector::*;

use std::marker::PhantomData;
use crate::WithDType;

#[derive(Copy, Clone)]
pub struct StorageViewUsf<T>(pub(crate) *mut T);

#[derive(Copy, Clone)]
pub struct MatrixViewUsf<'a, T> {
    storage: StorageViewUsf<T>,
    shape: (usize, usize),
    strides: (usize, usize),
    _marker: PhantomData<&'a mut T>, 
} 

#[derive(Copy, Clone)]
pub struct VectorViewUsf<'a, T> {
    storage: StorageViewUsf<T>,
    len: usize,
    stride: usize,
    _marker: PhantomData<&'a mut T>, 
}

impl<'a, T: WithDType> StorageViewUsf<T> {
    #[inline]
    pub unsafe fn get(&self, storage_index: usize) -> T {
        let ptr = unsafe { self.0.add(storage_index) };
        let value = unsafe { *ptr };
        value
    }

    #[inline]
    pub unsafe fn get_ref(&self, storage_index: usize) -> &T {
        let ptr = unsafe { self.0.add(storage_index) };
        unsafe { &*ptr }
    }

    #[inline]
    pub unsafe fn get_mut(&mut self, storage_index: usize) -> &mut T {
        let ptr = unsafe { self.0.add(storage_index) };
        unsafe { &mut *ptr }
    }

    #[inline]
    pub unsafe fn set(&mut self, storage_index: usize, value: T) {
        let ptr = unsafe { self.0.add(storage_index) };
        unsafe { *ptr = value };
    }

    #[inline]
    pub unsafe fn add(&self, storage_index: usize) -> *mut T {
        unsafe { self.0.add(storage_index) }
    }
}

#[cfg(test)]
mod tests {
    use crate::{view::MatrixViewUsf, NdArray, Result};
    use super::VectorViewUsf;

    #[test]
    fn test_from_ndarray_and_index() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3, 4])?;
        let view = VectorViewUsf::from_ndarray(&a)?;

        assert_eq!(view.len(), 4);
        assert_eq!(view[0], 1);
        assert_eq!(view[1], 2);
        assert_eq!(view[2], 3);
        assert_eq!(view[3], 4);
        Ok(())
    }

    #[test]
    fn test_set_and_get() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3])?;
        let mut view = VectorViewUsf::from_ndarray(&a)?;
        unsafe {
            view.set(1, 10);
            assert_eq!(view.g(1), 10);
        }
        Ok(())
    }

    #[test]
    fn test_take_and_drop() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3, 4, 5])?;
        let view = VectorViewUsf::from_ndarray(&a)?;

        let take2 = view.take(2)?;
        assert_eq!(take2.len(), 2);
        assert_eq!(take2[0], 1);
        assert_eq!(take2[1], 2);

        let drop2 = unsafe { view.drop(2)? };
        assert_eq!(drop2.len(), 3);
        assert_eq!(drop2[0], 3);
        assert_eq!(drop2[2], 5);

        Ok(())
    }

    #[test]
    fn test_swap() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3])?;
        let b = NdArray::new(&[4i32, 5, 6])?;

        let mut va = VectorViewUsf::from_ndarray(&a)?;
        let mut vb = VectorViewUsf::from_ndarray(&b)?;

        va.swap(&mut vb)?;
        assert_eq!(va[0], 4);
        assert_eq!(vb[0], 1);
        Ok(())
    }

    #[test]
    fn test_dot_and_norm() -> Result<()> {
        let a = NdArray::new(&[1f32, 2.0, 3.0])?;
        let b = NdArray::new(&[4f32, -5.0, 6.0])?;

        let va = VectorViewUsf::from_ndarray(&a)?;
        let vb = VectorViewUsf::from_ndarray(&b)?;

        let dot = unsafe { va.dot(&vb)? };
        assert!((dot - (1.0*4.0 + 2.0*-5.0 + 3.0*6.0)).abs() < 1e-6);

        let norm = unsafe { va.norm() };
        assert!((norm - (1f32*1.0 + 2.0*2.0 + 3.0*3.0).sqrt()).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_mul_assign_and_eqal() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3])?;
        let mut va = VectorViewUsf::from_ndarray(&a)?;

        unsafe { va.mul_assign(2); }
        assert_eq!(va[0], 2);
        assert_eq!(va[1], 4);
        assert_eq!(va[2], 6);

        let b = NdArray::new(&[2i32, 4, 6])?;
        let vb = VectorViewUsf::from_ndarray(&b)?;
        assert!(unsafe { va.eqal(&vb) });

        Ok(())
    }

    #[test]
    fn test_copy() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3])?;
        let view = VectorViewUsf::from_ndarray(&a)?;

        let b = unsafe { view.copy() };
        assert!(b.allclose(&a, 1e-6, 1e-6));
        Ok(())
    }

    #[test]
    fn test_from_ndarray_and_index_mat() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3, 4])?.reshape((2, 2))?;
        let view = MatrixViewUsf::from_ndarray(&a)?;

        assert_eq!(view.shape(), (2, 2));
        assert_eq!(view[(0, 0)], 1);
        assert_eq!(view[(0, 1)], 2);
        assert_eq!(view[(1, 0)], 3);
        assert_eq!(view[(1, 1)], 4);
        Ok(())
    }

    #[test]
    fn test_set_and_get_mat() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3, 4])?.reshape((2, 2))?;
        let mut view = MatrixViewUsf::from_ndarray(&a)?;

        unsafe { view.set(0, 1, 10); }
        assert_eq!(view[(0, 1)], 10);
        Ok(())
    }

    #[test]
    fn test_row_and_col() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3, 4, 5, 6])?.reshape((2, 3))?;
        let view = MatrixViewUsf::from_ndarray(&a)?;

        let row0 = unsafe { view.row(0)? };
        assert_eq!(row0.len(), 3);
        assert_eq!(row0[0], 1);
        assert_eq!(row0[2], 3);

        let col1 = unsafe { view.col(1)? };
        assert_eq!(col1.len(), 2);
        assert_eq!(col1[0], 2);
        assert_eq!(col1[1], 5);
        Ok(())
    }

    #[test]
    fn test_slice_and_copy() -> Result<()> {
        let a = NdArray::new(&[
            1i32, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ])?.reshape((3, 3))?;
        let view = MatrixViewUsf::from_ndarray(&a)?;

        // 取 1..3 行, 0..2 列
        let sub = unsafe { view.slice(1..3, 0..2)? };
        assert_eq!(sub.shape(), (2, 2));
        assert_eq!(sub[(0, 0)], 4);
        assert_eq!(sub[(1, 1)], 8);

        let copy = unsafe { sub.copy() };
        assert!(unsafe { sub.eqal(&MatrixViewUsf::from_ndarray(&copy)?) });
        Ok(())
    }

    #[test]
    fn test_copy_from() -> Result<()> {
        let a = NdArray::<i32>::zeros((2, 2))?;
        let mut view_a = MatrixViewUsf::from_ndarray(&a)?;
        let b = NdArray::new(&[1i32, 2, 3, 4])?.reshape((2, 2))?;
        let view_b = MatrixViewUsf::from_ndarray(&b)?;

        unsafe { view_a.copy_from(&view_b)? };
        assert_eq!(view_a[(0, 0)], 1);
        assert_eq!(view_a[(1, 1)], 4);
        Ok(())
    }

    #[test]
    fn test_transpose() -> Result<()> {
        let a = NdArray::new(&[1i32, 2, 3, 4, 5, 6])?.reshape((2, 3))?;
        let view = MatrixViewUsf::from_ndarray(&a)?;

        let t = view.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t[(0, 1)], 4); // transpose 位置
        Ok(())
    }

    #[test]
    fn test_swap_rows_and_cols() -> Result<()> {
        let a = NdArray::new(&[
            1i32, 2,
            3, 4,
        ])?.reshape((2, 2))?;
        let mut view = MatrixViewUsf::from_ndarray(&a)?;

        unsafe { view.swap_rows(0, 1)? };
        assert_eq!(view[(0, 0)], 3);
        assert_eq!(view[(1, 0)], 1);

        view.swap_cols(0, 1)?;
        assert_eq!(view[(0, 0)], 4);
        assert_eq!(view[(1, 1)], 1);
        Ok(())
    }

    #[test]
    fn test_iter_and_diag() -> Result<()> {
        let a = NdArray::new(&[
            1i32, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ])?.reshape((3, 3))?;
        let view = MatrixViewUsf::from_ndarray(&a)?;

        let elems: Vec<_> = unsafe { view.iter().collect() };
        assert_eq!(elems, [1,2,3,4,5,6,7,8,9]);

        let diag: Vec<_> = unsafe { view.diag().collect() };
        assert_eq!(diag, [1,5,9]);
        Ok(())
    }

    #[test]
    fn test_norm_and_matmul() -> Result<()> {
        let a = NdArray::new(&[1f32, 2.0, 3.0, 4.0])?.reshape((2, 2))?;
        let b = NdArray::new(&[5f32, 6.0, 7.0, 8.0])?.reshape((2, 2))?;

        let va = MatrixViewUsf::from_ndarray(&a)?;
        let vb = MatrixViewUsf::from_ndarray(&b)?;

        let norm = unsafe { va.norm() };
        assert!((norm - (1f32*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0).sqrt()).abs() < 1e-6);

        let c = unsafe { va.matmul(&vb)? };
        let expected = NdArray::new(&[19f32, 22.0, 43.0, 50.0])?.reshape((2, 2))?;
        assert!(c.allclose(&expected, 1e-6, 1e-6));
        Ok(())
    }
}
