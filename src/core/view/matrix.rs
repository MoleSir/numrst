use std::sync::{Arc, RwLock};

use crate::{Dim, Error, NdArray, Result, Storage, WithDType};

use super::{Vector, VectorLayout};

pub struct MatrixLayout {
    pub(crate) shape: (usize, usize),
    pub(crate)strides: (usize, usize),
    pub(crate)start_offset: usize,
}

pub struct Matrix<T: WithDType> {
    pub(crate)storage: Arc<RwLock<Storage<T>>>,
    pub(crate)layout: MatrixLayout,
}

impl<T: WithDType> Matrix<T> {
    pub fn from_ndarray(array: &NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;
        Self::from_ndarray_impl(array, 0, 1)
    }

    pub fn from_ndarray_axis<D1: Dim, D2: Dim>(array: &NdArray<T>, axis1: D1, axis2: D2) -> Result<Self> {
        let axis1 = axis1.to_index(array.shape(), "from_ndarray_axis")?;
        let axis2 = axis2.to_index(array.shape(), "from_ndarray_axis")?;
        Self::from_ndarray_impl(array, axis1, axis2)
    }

    fn from_ndarray_impl(array: &NdArray<T>, axis1: usize, axis2: usize) -> Result<Self> {
        if axis1 == axis2 {
            return Err(Error::Msg("select same axis in matrix".into()));
        }
        let layout = MatrixLayout {
            shape: (array.layout().dims()[axis1], array.layout().dims()[axis2]),
            strides: (array.layout().stride()[axis1], array.layout().stride()[axis2]),
            start_offset: array.layout().start_offset(),
        };
        Ok(Self {
            storage: array.storage_clone(),
            layout,
        })
    }

    pub fn get(&self, row: usize, col: usize) -> Result<T> {
        let index = self.layout.get_storage_index(row, col)?;
        Ok(self.storage.read().unwrap().data()[index])
    }

    pub fn set(&self, row: usize, col: usize, value: T) -> Result<()> {
        let index = self.layout.get_storage_index(row, col)?;
        let mut storage = self.storage.write().unwrap();
        let data = storage.data_mut(); 
        data[index] = value;
        Ok(())
    }

    pub fn row(&self, row: usize) -> Result<Vector<T>> {
        let layout = self.layout.get_row_vector_layout(row)?;
        let storage = self.storage.clone();
        Ok(Vector { storage, layout })
    }

    pub fn col(&self, col: usize) -> Result<Vector<T>> {
        let layout = self.layout.get_col_vector_layout(col)?;
        let storage = self.storage.clone();
        Ok(Vector { storage, layout })
    }
}

impl MatrixLayout {
    fn row_size(&self) -> usize {
        self.shape.0 
    }

    fn col_size(&self) -> usize {
        self.shape.1 
    }

    fn row_stride(&self) -> usize {
        self.strides.0 
    }

    fn col_stride(&self) -> usize {
        self.strides.1 
    }

    fn get_storage_index(&self, row: usize, col: usize) -> Result<usize> {
        if row >= self.row_size() || col >= self.col_size() {
            Err(Error::Msg("Matrix index out of range".into()))?
        }
        Ok(self.start_offset + self.row_stride() * row + self.col_stride() * col)
    }

    fn get_row_vector_layout(&self, row: usize) -> Result<VectorLayout> {
        if row > self.row_size() {
            return Err(Error::Msg("row index out of range".into()));
        } else {
            Ok(VectorLayout {
                len: self.col_size(),
                stride: self.col_stride(),
                start_offset: self.start_offset + row * self.row_stride(),
            })
        }
    }

    fn get_col_vector_layout(&self, col: usize) -> Result<VectorLayout> {
        if col > self.col_size() {
            return Err(Error::Msg("col index out of range".into()));
        } else {
            Ok(VectorLayout {
                len: self.row_size(),
                stride: self.row_stride(),
                start_offset: self.start_offset + col * self.col_stride(),
            })
        }
    }
}