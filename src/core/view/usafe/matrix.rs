use crate::{Error, FloatDType, Indexer, NdArray, NumDType, Result, Storage, WithDType};
use std::marker::PhantomData;
use super::{VectorViewUsf, MatrixViewUsf, StorageViewUsf};

impl<'a, T: WithDType> MatrixViewUsf<'a, T> {
    pub fn from_ndarray(array: &NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;

        Ok(Self {
            storage: StorageViewUsf(array.storage_ptr(array.layout().start_offset())),
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

    pub unsafe fn g_mut(&mut self, row: usize, col: usize) -> &mut T {
        let storage_index = self.storage_index(row, col);
        unsafe { self.storage.get_mut(storage_index) }
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
                storage: unsafe { StorageViewUsf(self.storage.add(row * self.row_stride())) },
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
                storage: unsafe { StorageViewUsf(self.storage.add(col * self.col_stride())) },
                len: self.row_size(),
                stride: self.row_stride(),
                _marker: PhantomData
            })
        }
    }

    pub unsafe fn slice<RI, CI>(&self, ri: RI, ci: CI) -> Result<Self>
    where
        RI: Into<Indexer>,
        CI: Into<Indexer>,
    {
        let ri: Indexer = ri.into();
        let ci: Indexer = ci.into();
        
        let (rbegin, rstep, rsize) = ri.begin_step_size(self.row_size(), "slice_row")?;
        let (cbegin, cstep, csize) = ci.begin_step_size(self.col_size(), "slice_col")?;

        let new_storage = StorageViewUsf(
            unsafe { self.storage.0.add(rbegin * self.row_stride() + cbegin * self.col_stride()) }
        );
        // 构造新的 view
        Ok(Self {
            storage: new_storage,
            shape: (rsize, csize),
            strides: (self.row_stride() * rstep, self.col_stride() * cstep),
            _marker: PhantomData,
        })
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

    pub unsafe fn copy_from(&mut self, source: &Self) -> Result<()> {
        if self.shape() != source.shape() {
            Err(Error::ShapeMismatchCopyFrom { dst: self.shape().into(), src: source.shape().into() })?
        }

        for r in 0..self.row_size() {
            for c in 0..self.col_size() {
                self[(r, c)] = source[(r, c)];
            }
        }

        Ok(())
    }

    pub fn transpose(&self) -> Self {
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

    pub unsafe fn swap_rows_partial(&mut self, r1: usize, r2: usize, size: usize) -> Result<()> {
        if r1 == r2 {
            return Ok(());
        }

        let mut row1 = unsafe { self.row(r1)?.take(size)? };
        let mut row2 = unsafe { self.row(r2)?.take(size)? };
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

    pub unsafe fn eqal(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            false 
        } else {
            unsafe {
                self.iter().zip(other.iter()).all(|(a, b)| a == b)
            }
        }
    }

    pub unsafe fn iter(&self) -> MatrixViewUsfIter<'_, T> {
        MatrixViewUsfIter { 
            view: self.clone(), 
            row: 0,
            col: 0,
        }
    }

    pub unsafe fn diag(&self) -> MatrixDiagIter<'_, T> {
        MatrixDiagIter {
            view: self.clone(),
            len: self.row_size().min(self.col_size()),
            index: 0
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
        let v =  unsafe { self.iter().map(|v| v.powi(2)).sum::<T>() };
        v.sqrt()
    }
}

impl<'a, T: NumDType> MatrixViewUsf<'a, T> {
    pub unsafe fn matmul(&self, rhs: &Self) -> Result<NdArray<T>> {
        let m = self.row_size();
        let k = self.col_size();
        let n = rhs.col_size();

        if k != rhs.row_size() {
            return Err(Error::ShapeMismatchBinaryOp { lhs: self.shape().into(), rhs: rhs.shape().into(), op: "matmul" })
        }

        let mut data = vec![T::zero(); m * n];
        // let mut result_storage = Storage::new(data);

        let mut result_view = MatrixViewUsf {
            storage: StorageViewUsf(data.as_mut_ptr()), 
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

pub struct MatrixViewUsfIter<'a, T: WithDType> {
    view: MatrixViewUsf<'a, T>,
    row: usize,
    col: usize,
}

impl<'a, T: WithDType> Iterator for MatrixViewUsfIter<'a, T> {
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

pub struct MatrixDiagIter<'a, T: WithDType> {
    view: MatrixViewUsf<'a, T>,
    len: usize,
    index: usize,
}

impl<'a, T: WithDType> Iterator for MatrixDiagIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.len {
            None
        } else {
            let v = unsafe { self.view.g(self.index, self.index) };
            self.index += 1;
            Some(v)
        }
    }
}