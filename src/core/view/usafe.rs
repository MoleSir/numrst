use crate::{Error, FloatDType, NdArray, NumDType, Result, Storage, WithDType};
use std::marker::PhantomData;

#[derive(Copy, Clone)]
pub struct StorageUsf<T>(*mut T);

#[derive(Copy, Clone)]
pub struct MatrixViewUsf<'a, T> {
    storage: StorageUsf<T>,
    shape: (usize, usize),
    strides: (usize, usize),
    _marker: PhantomData<&'a mut T>, 
} 

#[derive(Copy, Clone)]
pub struct VectorViewUsf<'a, T> {
    storage: StorageUsf<T>,
    len: usize,
    stride: usize,
    _marker: PhantomData<&'a mut T>, 
}

impl<'a, T: WithDType> StorageUsf<T> {
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

impl<'a, T: WithDType> MatrixViewUsf<'a, T> {
    pub fn from_ndarray(array: &NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;

        Ok(Self {
            storage: StorageUsf(array.storage_ptr(array.layout().start_offset())),
            shape: (array.layout().dims()[0], array.layout().dims()[1]),
            strides: (array.layout().stride()[0], array.layout().stride()[1]),
            _marker: PhantomData,
        })
    }

    pub fn is_square(&self) -> bool {
        let (m, n) = self.shape();
        m == n 
    }

    pub unsafe fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            Some(unsafe { self.g(row, col) })
        }
    }

    pub unsafe fn g(&self, row: usize, col: usize) -> T {
        let storage_index = self.storage_index(row, col);
        unsafe { self.storage.get(storage_index) }
    }

    pub unsafe fn set(&mut self, row: usize, col: usize, value: T) -> Option<()> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            Some(unsafe { self.s(row, col, value) })
        }
    }

    pub unsafe fn s(&mut self, row: usize, col: usize, value: T) {
        let storage_index = self.storage_index(row, col);
        let ptr = unsafe { self.storage.add(storage_index) };
        unsafe { *ptr = value };
    }

    pub unsafe fn row(&self, row: usize) -> Result<VectorViewUsf<'_, T>> {
        if row >= self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "row", len: self.row_size(), index: row });
        } else {
            Ok(VectorViewUsf {
                storage: unsafe { StorageUsf(self.storage.add(row * self.row_stride())) },
                len: self.col_size(),
                stride: self.col_stride(),
                _marker: PhantomData
            })
        }
    }

    pub unsafe fn col(&self, col: usize) -> Result<VectorViewUsf<'_, T>> {
        if col >= self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "col", len: self.col_size(), index: col });
        } else {
            Ok(VectorViewUsf {
                storage: unsafe { StorageUsf(self.storage.add(col * self.col_stride())) },
                len: self.row_size(),
                stride: self.row_stride(),
                _marker: PhantomData
            })
        }
    }

    pub unsafe fn copy(&self) -> NdArray<T> {
        let data: Vec<_> = (0..self.row_size())
            .flat_map(|row| {
                (0..self.col_size()).map(move |col| unsafe {
                    self.g(row, col)
                })
            })
            .collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, (self.row_size(), self.col_size()))
    }

    pub fn transpose(&'a self) -> Self {
        Self {
            shape: (self.col_size(), self.row_size()),
            strides: (self.col_stride(), self.row_stride()),
            storage: self.storage.clone(),
            _marker: PhantomData
        }
    }

    pub unsafe fn swap_rows(&mut self, r1: usize, r2: usize) -> Result<()> {
        if r1 == r2 {
            return Ok(());
        }

        let mut row1 = unsafe { self.row(r1)? };
        let mut row2 = unsafe { self.row(r2)? };
        row1.swap(&mut row2)
    }

    pub fn swap_cols(&mut self, c1: usize, c2: usize) -> Result<()> {
        if c1 == c2 {
            return Ok(());
        }

        let mut col1 = unsafe { self.col(c1)? };
        let mut col2 = unsafe { self.col(c2)? };
        col1.swap(&mut col2)
    }

    pub fn iter(&'a self) -> MatrixIterUsf<'a, T> {
        MatrixIterUsf { 
            view: self.clone(), 
            row: 0,
            col: 0,
        }
    }

    #[inline]
    pub fn element_size(&self) -> usize {
        self.row_size() * self.col_size()
    }

    #[inline]
    pub fn row_size(&self) -> usize {
        self.shape.0
    }

    #[inline]
    pub fn col_size(&self) -> usize {
        self.shape.1
    }

    #[inline]
    pub fn row_stride(&self) -> usize {
        self.strides.0
    }

    #[inline]
    pub fn col_stride(&self) -> usize {
        self.strides.1
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.row_size(), self.col_size())
    }

    #[inline]
    pub fn storage_index(&self, row: usize, col: usize) -> usize {
        self.strides.0 * row + self.strides.1 * col
    }
}

impl<'a, T: FloatDType> MatrixViewUsf<'a, T> {
    pub unsafe fn norm(&self) -> T {
        let v = self.iter().map(|v| v.powi(2)).sum::<T>();
        v.sqrt()
    }
}

impl<'a, T: NumDType> MatrixViewUsf<'a, T> {
    pub unsafe fn matmul(&self, rhs: &Self) -> Result<NdArray<T>> {
        let m = self.row_size();
        let k = self.col_size();
        let n = rhs.col_size();

        if k != rhs.row_size() {
            return Err(Error::Msg(format!(
                "Matrix dimension mismatch: self is {}x{}, rhs is {}x{}",
                m, k, rhs.row_size(), n
            )));
        }

        let mut data = vec![T::zero(); m * n];
        // let mut result_storage = Storage::new(data);

        let mut result_view = MatrixViewUsf {
            storage: StorageUsf(data.as_mut_ptr()), 
            shape: (m, n),
            strides: (n, 1), // row-major
            _marker: PhantomData,
        };

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for kk in 0..k {
                    let a = unsafe { self.g(i, kk) }; 
                    let b = unsafe { rhs.g(kk, j) };
                    sum = sum + a * b;
                }
                unsafe { result_view.s(i, j, sum); }
            }
        }

        let storage = Storage::new(data);
        Ok(NdArray::from_storage(storage, (m, n)))
    }
}

//////////////////////////////////////////////////////////////////////////////////////

impl<'a, T: WithDType> VectorViewUsf<'a, T> {
    pub fn from_ndarray(array: &NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;

        Ok(Self {
            storage: StorageUsf(array.storage_ptr(array.layout().start_offset())),
            len: array.layout().dims()[0],
            stride: array.layout().stride()[0],
            _marker: PhantomData,
        })
    }

    pub unsafe fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            None
        } else {
            Some(unsafe { self.g(index) })
        }
    }

    pub unsafe fn g(&self, index: usize) -> T {
        let storage_index = self.storage_index(index);
        unsafe { self.storage.get(storage_index) }
    }

    pub unsafe fn set(&mut self, index: usize, value: T) -> Option<()> {
        if index >= self.len {
            None
        } else {
            unsafe { self.s(index, value); }
            Some(())
        }
    }

    pub unsafe fn s(&mut self, index: usize, value: T) {
        let storage_index = self.storage_index(index);
        let ptr = unsafe { self.storage.add(storage_index) };
        unsafe { *ptr = value };
    }

    pub unsafe fn copy(&self) -> NdArray<T> {
        let data: Vec<_> = (0..self.len)
            .map(|index| unsafe { self.g(index) })
            .collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, (self.len,))
    }

    pub fn swap(&mut self, other: &mut Self) -> Result<()> {
        if self.len() != other.len() {
            return Err(Error::Msg("len mismatch".into()))?;
        }

        let stride = self.stride;
        let storage_index = |index: usize| {
            stride * index
        };

        let len = self.len();
        (0..len).into_iter()
            .map(|index| (storage_index(index), storage_index(index)))
            .for_each(|(self_index, other_index)| {
                unsafe {
                    let self_value = self.storage.get(self_index);
                    let other_value = other.storage.get(other_index);
                    self.storage.set(self_index, other_value);
                    other.storage.set(other_index, self_value);
                }
            });

        Ok(())
    }

    pub fn iter(&'a self) -> VectorIterUsf<'a, T> {
        VectorIterUsf { 
            view: self.clone(), 
            index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn storage_index(&self, index: usize) -> usize {
        self.stride * index
    }
}

impl<'a, T: FloatDType> VectorViewUsf<'a, T> {
    pub unsafe fn norm(&self) -> T {
        let v = self.iter().map(|v| v.powi(2)).sum::<T>();
        v.sqrt()
    }
}

impl<'a, T: NumDType> VectorViewUsf<'a, T> {
    pub unsafe fn dot(&self, rhs: &Self) -> Result<T> {
        if self.len() != rhs.len() {
            Err(Error::Msg("mis match size".into()))?;
        }
        Ok(self.iter().zip(rhs.iter())
            .map(|(a, b)| a * b)
            .sum::<T>())
    }

    pub unsafe fn mul_assign(&'a mut self, mul: T) {
        for i in 0..self.len {
            let v = unsafe { self.g(i) };
            unsafe { self.s(i, v * mul); }
        }
    }
}

/////////////////////////////////////////////////////////////////////

pub struct MatrixIterUsf<'a, T: WithDType> {
    view: MatrixViewUsf<'a, T>,
    row: usize,
    col: usize,
}

impl<'a, T: WithDType> Iterator for MatrixIterUsf<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.col == self.view.col_size() {
            self.col = 0;
            self.row += 1;
            if self.row == self.view.row_size() {
                return None;
            } 
        }

        let value = unsafe { self.view.g(self.row, self.col) };
        self.col += 1;
        Some(value)
    }
}

pub struct VectorIterUsf<'a, T: WithDType> {
    view: VectorViewUsf<'a, T>,
    index: usize,
}

impl<'a, T: WithDType> Iterator for VectorIterUsf<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.len {
            None
        } else {
            let value = unsafe { self.view.g(self.index) };
            self.index += 1;
            Some(value)
        }
    }
}

impl<'a, T: WithDType> std::ops::Index<(usize, usize)> for MatrixViewUsf<'a, T> {
    type Output = T;
    fn index(&self, (row, col): (usize, usize)) -> &T {
        let storage_index = self.storage_index(row, col);
        unsafe { self.storage.get_ref(storage_index) }
    }
}

impl<'a, T: WithDType> std::ops::IndexMut<(usize, usize)> for MatrixViewUsf<'a, T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        let storage_index = self.storage_index(row, col);
        unsafe { self.storage.get_mut(storage_index) }
    }
}

impl<'a, T: WithDType> std::ops::Index<usize> for VectorViewUsf<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        let storage_index = self.storage_index(index);
        unsafe { self.storage.get_ref(storage_index) }
    }
}

impl<'a, T: WithDType> std::ops::IndexMut<usize> for VectorViewUsf<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        let storage_index = self.storage_index(index);
        unsafe { self.storage.get_mut(storage_index) }
    }
}
