mod usafe;
mod matrix;
mod vector;

use std::marker::PhantomData;

pub use usafe::*;
pub use matrix::*;
pub use vector::*;

use crate::{Error, Result};
use super::{FloatDType, NdArray, NumDType, Storage, WithDType};

////////////////////////////////////////////////////////////////////////////
///     AsVector
////////////////////////////////////////////////////////////////////////////

pub trait AsVectorView<'a, T: WithDType> : Sized {
    fn len(&self) -> usize;
    fn stride(&self) -> usize;
    fn storage_get_uncheck(&self, storage_index: usize) -> T;
    fn from_ndarray(arr: &'a NdArray<T>) -> Result<Self>;

    fn get(&self, index: usize) -> Option<T> {
        if index >= self.len() {
            None
        } else {
            Some(self.storage_get_uncheck(self.storage_index(index)))
        }
    }

    #[inline]
    fn g(&self, index: usize) -> T {
        self.storage_get_uncheck(self.storage_index(index))
    }

    fn eqal(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            false 
        } else {
            for i in 0..self.len() {
                if self.g(i) != other.g(i) {
                    return false;
                }
            }
            true
        }
    }

    fn copy(&self) -> NdArray<T> {
        let data: Vec<_> = (0..self.len())
            .map(|index| self.g(index))
            .collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, self.len())
    }

    fn iter(&'a self) -> VectorViewIter<'a, T, Self> {
        VectorViewIter { 
            view: self, 
            index: 0,
            _marker: PhantomData,
        }
    }

    fn to_vec(&'a self) -> Vec<T> {
        self.iter().collect()
    }

    #[inline]
    fn storage_index(&self, index: usize) -> usize {
        self.stride() * index
    }
}

pub trait AsVectorViewNum<'a, T: NumDType> : AsVectorView<'a, T> {
    fn dot(&self, rhs: &Self) -> Result<T> {
        if self.len() != rhs.len() {
            Err(Error::Msg("mis match size".into()))?;
        }

        let v = (0..self.len())
            .map(|index| self.g(index) * rhs.g(index))
            .sum::<T>();

        Ok(v)
    }
}

pub trait AsVectorViewFloat<'a, T: FloatDType> : AsVectorViewNum<'a, T> {
    fn norm(&'a self) -> T {
        self.iter()
            .map(|v| v.powi(2))
            .sum::<T>()
            .sqrt()
    }
}

pub trait AsVectorViewMut<'a, T: WithDType> : AsVectorView<'a, T> {
    fn storage_set_uncheck(&mut self, storage_index: usize, value: T);
    fn from_ndarray_mut(arr: &'a mut NdArray<T>) -> Result<Self>;

    fn set(&mut self, index: usize, value: T) -> Option<()> {
        if index >= self.len() {
            None
        } else {
            self.storage_set_uncheck(self.storage_index(index), value);
            Some(())
        }
    }

    #[inline]
    fn s(&mut self, index: usize, value: T) {
        self.storage_set_uncheck(self.storage_index(index), value);
    }

    fn swap(&mut self, other: &mut Self) -> Result<()> {
        if self.len() != other.len() {
            return Err(Error::Msg("len mismatch".into()));
        }
    
        let self_stride = self.stride();
        let other_stride = other.stride();
    
        for i in 0..self.len() {
            let si = self_stride * i;
            let oi = other_stride * i;
    
            let a = self.storage_get_uncheck(si);
            let b = other.storage_get_uncheck(oi);
    
            self.storage_set_uncheck(si, b);
            other.storage_set_uncheck(oi, a);
        }
    
        Ok(())
    }

    fn copy_from(&'a mut self, arr: &NdArray<T>) -> Result<()> {
        let view = arr.vector_view()?;
        self.copy_from_view(&view)
    }

    fn copy_from_view<'b>(&mut self, source: &impl AsVectorView<'b, T>) -> Result<()> {
        if self.len() != source.len() {
            Err(Error::LenMismatchVector { lhs: self.len(), rhs: source.len(), op: "copy_from_view" })?;
        }

        for i in 0..self.len() {
            self.s(i, source.g(i));
        }

        Ok(())
    }
}

pub trait AsVectorViewMutNum<'a, T: NumDType> : AsVectorViewMut<'a, T> {
    fn assign_op<F>(&mut self, rhs: T, f: F) 
    where 
        F: Fn(T, T) -> T
    {
        for i in 0..self.len() {
            let v = self.g(i);
            self.s(i, f(v, rhs));
        }
    } 

    #[inline]
    fn add_assign(&mut self, add: T) {
        self.assign_op(add, |a, b| a + b);
    }

    fn sub_assign(&mut self, sub: T) {
        self.assign_op(sub, |a, b| a - b);
    }

    fn mul_assign(&mut self, mul: T) {
        self.assign_op(mul, |a, b| a * b);
    }

    fn div_assign(&mut self, div: T) {
        self.assign_op(div, |a, b| a / b);
    }
}

pub struct VectorViewIter<'a, T: WithDType, V: AsVectorView<'a, T>> {
    view: &'a V,
    index: usize,
    _marker: PhantomData<T>,
}

impl<'a, T: WithDType, V: AsVectorView<'a, T>> Iterator for VectorViewIter<'a, T, V> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.len() {
            None
        } else {
            let value = self.view.g(self.index);
            self.index += 1;
            Some(value)
        }
    }
}

impl<'a, T: WithDType, V: AsVectorView<'a, T>> ExactSizeIterator for VectorViewIter<'a, T, V> {
    fn len(&self) -> usize {
        self.view.len()
    }
}

////////////////////////////////////////////////////////////////////////////
///     AsMatrix
////////////////////////////////////////////////////////////////////////////

pub trait AsMatrixView<'a, T: WithDType> : Sized {
    fn shape(&self) -> (usize, usize);
    fn strides(&self) -> (usize, usize);
    fn storage_get_uncheck(&self, storage_index: usize) -> T;
    fn from_ndarray(arr: &'a NdArray<T>) -> Result<Self>;

    #[inline]
    fn row_size(&self) -> usize {
        self.shape().0
    }

    #[inline]
    fn col_size(&self) -> usize {
        self.shape().1
    }

    #[inline]
    fn row_stride(&self) -> usize {
        self.strides().0
    }

    #[inline]
    fn col_stride(&self) -> usize {
        self.strides().1
    }

    fn is_square(&self) -> bool {
        let (m, n) = self.shape();
        m == n 
    }

    #[inline]
    fn storage_index(&self, row: usize, col: usize) -> usize {
        self.row_stride() * row + self.col_stride() * col
    }

    fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            Some(self.storage_get_uncheck(self.storage_index(row, col)))
        }
    }

    #[inline]
    fn g(&self, row: usize, col: usize) -> T {
        self.storage_get_uncheck(self.storage_index(row, col))
    }

    fn eqal(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            false 
        } else {
            for i in 0..self.row_size() {
                for j in 0..self.col_size() {
                    if self.g(i, j) != other.g(i, j) {
                        return false;
                    }
                }
            }
            true
        }
    }

    fn copy(&'a self) -> NdArray<T> {
        let data: Vec<_> = self.iter().collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, (self.row_size(), self.col_size()))
    }

    fn diag(&self) -> Vec<T> {
        let m = self.row_size().min(self.col_size());
        (0..m)
            .map(|i| self.g(i, i))
            .collect()
    }

    fn iter(&'a self) -> MatrixViewIter<'a, T, Self> {
        MatrixViewIter {
            _marker: PhantomData,
            view: self, 
            row: 0,
            col: 0,
        }
    }

    fn to_vec(&'a self) -> Vec<T> {
        self.iter().collect()
    }
}

pub trait AsMatrixViewNum<'a, T: NumDType> : AsMatrixView<'a, T> {
    fn matmul(&self, rhs: &Self) -> Result<NdArray<T>> {
        let m = self.row_size();
        let k = self.col_size();
        let n = rhs.col_size();

        if k != rhs.row_size() {
            return Err(Error::ShapeMismatchBinaryOp { lhs: self.shape().into(), rhs: rhs.shape().into(), op: "matmul" })
        }

        let mut data = vec![T::zero(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for kk in 0..k {
                    let a = self.g(i, kk); 
                    let b = rhs.g(kk, j);
                    sum = sum + a * b;
                }
                data[i * self.col_size() + j] = sum;
            }
        }

        let storage = Storage::new(data);
        Ok(NdArray::from_storage(storage, (m, n)))
    }
}

pub trait AsMatrixViewFloat<'a, T: FloatDType> : AsMatrixView<'a, T> {
    fn norm(&'a self) -> T {
        self.iter()
            .map(|v| v.powi(2))
            .sum::<T>()
            .sqrt()
    }
}

pub trait AsMatrixViewMut<'a, T: WithDType> : AsMatrixView<'a, T> {
    fn storage_set_uncheck(&mut self, storage_index: usize, value: T);
    fn from_ndarray_mut(arr: &'a mut NdArray<T>) -> Result<Self>;

    fn set(&mut self, row: usize, col: usize, value: T) -> Option<()> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            self.storage_set_uncheck(self.storage_index(row, col), value);
            Some(())
        }
    }

    #[inline]
    fn s(&mut self, row: usize, col: usize, value: T) {
        self.storage_set_uncheck(self.storage_index(row, col), value)
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) -> Result<()> {
        if r1 == r2 {
            return Ok(());
        }
        if r1 > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_rows", len: self.row_size(), index: r1 });
        } 
        if r2 > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_rows", len: self.row_size(), index: r2 });
        } 

        let stride = self.col_stride();
        let mut start_index1 = self.col_size() * r1;
        let mut start_index2 = self.col_size() * r2;

        for _ in 0..self.col_size() {
            let value1 = self.storage_get_uncheck(start_index1);
            let value2 = self.storage_get_uncheck(start_index2);
            self.storage_set_uncheck(start_index1, value2);
            self.storage_set_uncheck(start_index2, value1);
            start_index1 += stride;
            start_index2 += stride;
        }

        Ok(())    
    }

    fn swap_cols(&mut self, c1: usize, c2: usize) -> Result<()> {
        if c1 == c2 {
            return Ok(());
        }
        if c1 > self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_cols", len: self.row_size(), index: c1 });
        } 
        if c2 > self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_cols", len: self.row_size(), index: c2 });
        } 

        let stride = self.row_stride();
        let mut start_index1 = c1;
        let mut start_index2 = c2;

        for _ in 0..self.row_size() {
            let value1 = self.storage_get_uncheck(start_index1);
            let value2 = self.storage_get_uncheck(start_index2);
            self.storage_set_uncheck(start_index1, value2);
            self.storage_set_uncheck(start_index2, value1);
            start_index1 += stride;
            start_index2 += stride;
        }

        Ok(())    
    }
    
    fn copy_from(&mut self, arr: &NdArray<T>) -> Result<()> {
        let view = arr.matrix_view()?;
        self.copy_from_view(&view)
    }
    
    fn copy_from_view<'b>(&mut self, source: &impl AsMatrixView<'b, T>) -> Result<()> {
        if self.shape() != source.shape() {
            return Err(Error::ShapeMismatchMatrix {
                lhs: self.shape(),
                rhs: source.shape(),
                op: "copy_from_view",
            });
        }
    
        for i in 0..self.row_size() {
            for j in 0..self.col_size() {
                self.s(i, j, source.g(i, j));
            }
        }
    
        Ok(())
    }
}

pub struct MatrixViewIter<'a, T: WithDType, V: AsMatrixView<'a, T>> {
    view: &'a V,
    row: usize,
    col: usize,
    _marker: PhantomData<T>,
}

impl<'a, T: WithDType, V: AsMatrixView<'a, T>> Iterator for MatrixViewIter<'a, T, V> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.col == self.view.col_size() {
            self.col = 0;
            self.row += 1;
            if self.row == self.view.row_size() {
                return None;
            } 
        }

        let value = self.view.g(self.row, self.col);
        self.col += 1;
        Some(value)
    }
}