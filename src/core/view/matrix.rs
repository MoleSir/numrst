use crate::{Error, NdArray, Result, Storage, StorageMut, WithDType, StorageRef};
use super::{VectorView, VectorViewMut};

pub struct MatrixView<'a, T: WithDType> {
    pub(crate) storage: StorageRef<'a, T>,
    pub(crate) shape: (usize, usize),
    pub(crate) strides: (usize, usize),
}

impl<'a, T: WithDType> MatrixView<'a, T> {
    pub fn from_ndarray(array: &'a NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;

        Ok(Self {
            storage: array.storage_ref(array.layout().start_offset()),
            shape: (array.layout().dims()[0], array.layout().dims()[1]),
            strides: (array.layout().stride()[0], array.layout().stride()[1]),
        })
    }

    pub fn clone(&'a self) -> Self {
        Self {
            shape: self.shape,
            storage: self.storage.clone(),
            strides: self.strides
        }
    }

    pub fn row(&'a self, row: usize) -> Result<VectorView<'a, T>> {
        if row > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "row", len: self.row_size(), index: row });
        } else {
            let storage = self.storage.slice(row * self.row_stride());
            Ok(VectorView {
                storage,
                len: self.col_size(),
                stride: self.col_stride(),
            })
        }
    }

    pub fn col(&'a self, col: usize) -> Result<VectorView<'a, T>> {
        if col > self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "col", len: self.col_size(), index: col });
        } else {
            let storage = self.storage.slice(col * self.col_stride());
            Ok(VectorView {
                storage,
                len: self.row_size(),
                stride: self.row_stride(),
            })
        }
    }

    pub fn is_square(&self) -> bool {
        let (m, n) = self.shape();
        m == n 
    }

    pub fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            Some(self.storage.get_unchecked(self.storage_index(row, col)))
        }
    }

    pub fn copy(&self) -> NdArray<T> {
        let data: Vec<_> = (0..self.row_size() * self.col_size()).into_iter()
            .map(|index| self.storage.get_unchecked(index * self.col_stride()))
            .collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, (self.row_size(), self.col_size()))
    }

    /// Uncheck `get`
    #[inline]
    pub fn g(&self, row: usize, col: usize) -> T {
        self.storage.get_unchecked(self.storage_index(row, col))
    }

    pub fn row_size(&self) -> usize {
        self.shape.0
    }

    pub fn col_size(&self) -> usize {
        self.shape.1
    }

    pub fn row_stride(&self) -> usize {
        self.strides.0
    }

    pub fn col_stride(&self) -> usize {
        self.strides.1
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.row_size(), self.col_size())
    }

    pub fn storage_index(&self, row: usize, col: usize) -> usize {
        self.strides.0 * row + self.strides.1 * col
    }
}

pub struct MatrixViewMut<'a, T: WithDType> {
    pub(crate) storage: StorageMut<'a, T>,
    pub(crate) shape: (usize, usize),
    pub(crate) strides: (usize, usize),
}

impl<'a, T: WithDType> MatrixViewMut<'a, T> {
    pub fn from_ndarray(array: &'a NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;

        Ok(Self {
            storage: array.storage_mut(array.layout().start_offset()),
            shape: (array.layout().dims()[0], array.layout().dims()[1]),
            strides: (array.layout().stride()[0], array.layout().stride()[1]),
        })
    }

    pub fn clone(&'a self) -> MatrixView<'a, T> {
        MatrixView {
            shape: self.shape,
            storage: self.storage.clone(),
            strides: self.strides
        }
    }

    pub fn clone_mut(&'a mut self) -> Self {
        Self {
            shape: self.shape,
            storage: self.storage.clone_mut(),
            strides: self.strides
        }
    }

    pub fn copy(&self) -> NdArray<T> {
        let data: Vec<_> = (0..self.row_size() * self.col_size()).into_iter()
            .map(|index| self.storage.get_unchecked(index * self.col_stride()))
            .collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, (self.row_size(), self.col_size()))
    }

    pub fn row(&'a self, row: usize) -> Result<VectorView<'a, T>> {
        if row > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "row", len: self.row_size(), index: row });
        } else {
            let storage = self.storage.slice(row * self.row_stride());
            Ok(VectorView {
                storage,
                len: self.col_size(),
                stride: self.col_stride(),
            })
        }
    }

    pub fn col(&'a self, col: usize) -> Result<VectorView<'a, T>> {
        if col > self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "col", len: self.col_size(), index: col });
        } else {
            let storage = self.storage.slice(col * self.col_stride());
            Ok(VectorView {
                storage,
                len: self.row_size(),
                stride: self.row_stride(),
            })
        }
    }

    pub fn row_mut(&'a mut self, row: usize) -> Result<VectorViewMut<'a, T>> {
        if row > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "row_mut", len: self.row_size(), index: row });
        } else {
            let len = self.col_size();
            let stride = self.col_stride();
            let storage = self.storage.slice_mut(row * self.row_stride());
            Ok(VectorViewMut {
                storage,
                len,
                stride,
            })
        }
    }

    pub fn col_mut(&'a mut self, col: usize) -> Result<VectorViewMut<'a, T>> {
        if col > self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "col_mut", len: self.col_size(), index: col });
        } else {
            let len = self.row_size();
            let stride = self.row_stride();
            let storage = self.storage.slice_mut(col * self.col_stride());
            Ok(VectorViewMut {
                storage,
                len,
                stride,
            })
        }
    }

    pub fn swap_rows(&mut self, r1: usize, r2: usize) -> Result<()> {
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
            let value1 = self.storage.get_unchecked(start_index1);
            let value2 = self.storage.get_unchecked(start_index2);
            self.storage.set_unchecked(start_index1, value2);
            self.storage.set_unchecked(start_index2, value1);
            start_index1 += stride;
            start_index2 += stride;
        }

        Ok(())    
    }

    pub fn swap_rows_partial(&mut self, r1: usize, r2: usize, size: usize) -> Result<()> {
        if r1 == r2 {
            return Ok(());
        }
        if r1 > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_rows_partial", len: self.row_size(), index: r1 });
        } 
        if r2 > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_rows_partial", len: self.row_size(), index: r2 });
        } 

        let stride = self.col_stride();
        let mut start_index1 = self.col_size() * r1;
        let mut start_index2 = self.col_size() * r2;

        for _ in 0..size {
            let value1 = self.storage.get_unchecked(start_index1);
            let value2 = self.storage.get_unchecked(start_index2);
            self.storage.set_unchecked(start_index1, value2);
            self.storage.set_unchecked(start_index2, value1);
            start_index1 += stride;
            start_index2 += stride;
        }

        Ok(())  
    }

    pub fn is_square(&self) -> bool {
        let (m, n) = self.shape();
        m == n 
    }

    pub fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            Some(self.storage.get_unchecked(self.storage_index(row, col)))
        }
    }

    /// Uncheck `get`
    #[inline]
    pub fn g(&self, row: usize, col: usize) -> T {
        self.storage.get_unchecked(self.storage_index(row, col))
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) -> Option<()> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            self.storage.set_unchecked(self.storage_index(row, col), value);
            Some(())
        }
    }

    /// Uncheck `get`
    #[inline]
    pub fn s(&mut self, row: usize, col: usize, value: T) {
        self.storage.set_unchecked(self.storage_index(row, col), value)
    }

    pub fn row_size(&self) -> usize {
        self.shape.0
    }

    pub fn col_size(&self) -> usize {
        self.shape.1
    }

    pub fn row_stride(&self) -> usize {
        self.strides.0
    }

    pub fn col_stride(&self) -> usize {
        self.strides.1
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.row_size(), self.col_size())
    }

    pub fn storage_index(&self, row: usize, col: usize) -> usize {
        self.strides.0 * row + self.strides.1 * col
    }
}
