mod construct;
mod indexer;
mod iter;
mod display;
mod shape;
mod arith;
mod matmul;
mod reduce;
mod broadcast;
mod convert;
mod condition;

use std::sync::{Arc, RwLock};
pub use indexer::{Range, IndexOp};
use crate::{Error, Result};
use super::{view::{Matrix, Vector}, DType, Dim, Layout, NumDType, Shape, Storage, WithDType};
pub use iter::*;

#[derive(Clone)]
pub struct NdArray<D>(Arc<NdArrayImpl<D>>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NdArrayId(usize);

impl NdArrayId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub struct NdArrayImpl<T> {
    id: NdArrayId,
    storage: Arc<RwLock<Storage<T>>>,
    layout: Layout,
}

impl<T: WithDType> NdArray<T> {
    pub fn is_scaler(&self) -> bool {
        self.shape().is_scaler()
    }

    pub fn to_scalar(&self) -> Result<T> {
        if !self.is_scaler() {
            Err(Error::Msg("not a scalar".into()))
        } else {
            let storage = self.0.storage.read().unwrap();
            let data = storage.data();
            let index = self.layout().start_offset();
            let scalar = data[index];
            Ok(scalar)
        }
    }

    pub fn set_scalar(&self, val: T) -> Result<()> {
        if !self.is_scaler() {
            Err(Error::Msg("not a scalar".into()))
        } else {
            let mut storage = self.0.storage.write().unwrap();
            let data = storage.data_mut();
            let index = self.layout().start_offset();
            data[index] = val;
            Ok(())
        }
    }
}

impl<T: WithDType> NdArray<T> {
    pub fn id(&self) -> usize {
        self.0.id.0
    }

    pub fn shape(&self) -> &Shape {
        self.0.layout.shape()
    }

    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn storage(&self) -> std::sync::RwLockReadGuard<'_, Storage<T>> {
        self.0.storage.read().unwrap()
    }

    pub fn element_count(&self) -> usize {
        self.shape().element_count()
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.iter().collect()
    }

    pub(crate) fn storage_clone(&self) -> Arc<RwLock<Storage<T>>> {
        self.0.storage.clone()
    }
}

impl<T: NumDType> NdArray<T> {
    pub fn matrix_view(&self) -> Result<Matrix<T>> {
        Matrix::<T>::from_ndarray(self)
    }

    pub fn vector_view(&self) -> Result<Vector<T>> {
        Vector::<T>::from_ndarray(self)
    }

    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a.close(b, rtol, atol))
    }
}
