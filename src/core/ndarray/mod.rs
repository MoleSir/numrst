mod create;
mod index;
mod binary_op;
mod unary_op;
mod matmul;
mod iter;
mod display;

use std::sync::{Arc, RwLock};
use super::{DType, Layout, Shape, Storage};

#[derive(Clone)]
pub struct NdArray(Arc<NdArrayImpl>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NdArrayId(usize);

impl NdArrayId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub struct NdArrayImpl {
    id: NdArrayId,
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    dtype: DType,
}

impl NdArray {
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

    pub fn storage(&self) -> std::sync::RwLockReadGuard<'_, Storage> {
        self.0.storage.read().unwrap()
    }

    pub fn element_count(&self) -> usize {
        self.shape().element_count()
    }

    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a.allclose(&b, rtol, atol))
    }
}