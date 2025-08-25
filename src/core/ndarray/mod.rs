mod construct;
mod indexer;
mod iter;
mod display;
mod shape;
mod ops;
mod matmul;
mod reduce;
mod broadcast;

use std::sync::{Arc, RwLock};
pub use indexer::{Range, IndexOp};
use crate::{Error, Result};
use super::{DType, Dim, Layout, Shape, Storage, WithDType};

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
    dtype: DType,
}

impl<T: WithDType> NdArray<T> {
    pub fn is_scaler(&self) -> bool {
        self.shape().is_scaler()
    }

    pub fn to_scalar(&self) -> Result<T> {
        if !self.is_scaler() {
            Err(Error::Msg("not a scalar".into()))
        } else {
            Ok(self.iter().next().unwrap())
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
        self.0.dtype
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

    pub fn stride(&self) -> &[usize] {
        self.layout().stride()
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

    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a.close(b, rtol, atol))
    }
}