use std::sync::{Arc, RwLock};

use crate::{Error, NdArray, Result, Storage, WithDType};

use super::{Matrix, MatrixLayout, Vector, VectorLayout};

pub struct CubeLayout {
    shape: (usize, usize, usize),
    strides: (usize, usize, usize),
    start_offset: usize,
}

pub struct Cube<T: WithDType> {
    storage: Arc<RwLock<Storage<T>>>,
    layout: CubeLayout,
}

impl<T: WithDType> Cube<T> {
    pub fn from_ndarray(array: &NdArray<T>) -> Result<Self> {
        let _ = array.dims3()?; // 确保是三维
        let layout = CubeLayout {
            shape: (
                array.layout().dims()[0],
                array.layout().dims()[1],
                array.layout().dims()[2],
            ),
            strides: (
                array.layout().stride()[0],
                array.layout().stride()[1],
                array.layout().stride()[2],
            ),
            start_offset: array.layout().start_offset(),
        };
        Ok(Self {
            storage: array.storage_clone(),
            layout,
        })
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> Result<T> {
        let index = self.layout.get_storage_index(i, j, k)?;
        Ok(self.storage.read().unwrap().data()[index])
    }

    pub fn set(&self, i: usize, j: usize, k: usize, value: T) -> Result<()> {
        let index = self.layout.get_storage_index(i, j, k)?;
        let mut storage = self.storage.write().unwrap();
        let data = storage.data_mut();
        data[index] = value;
        Ok(())
    }

    pub fn matrix(&self, i: usize) -> Result<Matrix<T>> {
        if i >= self.layout.shape.0 {
            return Err(Error::Msg("Cube matrix index out of range".into()));
        }
        let layout = MatrixLayout {
            shape: (self.layout.shape.1, self.layout.shape.2),
            strides: (self.layout.strides.1, self.layout.strides.2),
            start_offset: self.layout.start_offset + i * self.layout.strides.0,
        };
        Ok(Matrix {
            storage: self.storage.clone(),
            layout,
        })
    }

    pub fn vector(&self, i: usize, j: usize) -> Result<Vector<T>> {
        if i >= self.layout.shape.0 || j >= self.layout.shape.1 {
            return Err(Error::Msg("Cube vector index out of range".into()));
        }
        let layout = VectorLayout {
            len: self.layout.shape.2,
            stride: self.layout.strides.2,
            start_offset: self.layout.start_offset
                + i * self.layout.strides.0
                + j * self.layout.strides.1,
        };
        Ok(Vector {
            storage: self.storage.clone(),
            layout,
        })
    }
}

impl CubeLayout {
    fn get_storage_index(&self, i: usize, j: usize, k: usize) -> Result<usize> {
        if i >= self.shape.0 || j >= self.shape.1 || k >= self.shape.2 {
            return Err(Error::Msg("Cube index out of range".into()));
        }
        Ok(self.start_offset + i * self.strides.0 + j * self.strides.1 + k * self.strides.2)
    }
}
