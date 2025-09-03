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

use std::sync::Arc;
pub use indexer::{Range, IndexOp};
use crate::{Error, Result};
use super::{view::{AsMatrixView, AsMatrixViewMut, AsVectorView, AsVectorViewMut, MatrixView, MatrixViewMut, MatrixViewUsf, VectorView, VectorViewMut, VectorViewUsf}, DType, Dim, DimCoordinates, DimNCoordinates, Layout, NumDType, Shape, Storage, StorageArc, StorageIndices, StorageMut, StorageRef, WithDType};
pub use iter::*;
pub use indexer::*;

#[derive(Clone)]
pub struct NdArray<D>(pub(crate) Arc<NdArrayImpl<D>>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NdArrayId(usize);

impl NdArrayId {
    pub fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub struct NdArrayImpl<T> {
    pub(crate) id: NdArrayId,
    pub(crate) storage: StorageArc<T>,
    pub(crate) layout: Layout,
}

impl<T: WithDType> NdArray<T> {
    pub fn is_scalar(&self) -> bool {
        self.shape().is_scalar()
    }

    pub fn check_scalar(&self) -> Result<()> {
        if !self.is_scalar() {
            Err(Error::NotScalar)
        } else {
            Ok(())
        }
    }

    pub fn to_scalar(&self) -> Result<T> {
        self.check_scalar()?;
        let v = self.storage_ref(self.layout().start_offset()).get_unchecked(0);
        Ok(v)
    }

    pub fn set_scalar(&self, val: T) -> Result<()> {
        self.check_scalar()?;
        self.storage_mut(self.layout().start_offset()).set_unchecked(0, val);
        Ok(())
    }

    #[inline]
    pub fn storage_ref<'a>(&'a self, start_offset: usize) -> StorageRef<'a, T> {
        self.0.storage.get_ref(start_offset)
    }

    #[inline]
    pub fn storage_mut<'a>(&'a self, start_offset: usize) -> StorageMut<'a, T> {
        self.0.storage.get_mut(start_offset)
    }

    #[inline]
    pub fn storage_ptr(&self, start_offset: usize) -> *mut T {
        self.0.storage.get_ptr(start_offset)
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
        self.0.storage.0.read().unwrap()
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

    /// Returns an iterator over **storage indices**.
    ///
    /// This iterator yields the linear (flat) indices as they are laid out
    /// in the underlying storage buffer. The order depends on the memory
    /// layout (e.g., row-major / column-major / with strides).
    ///
    /// Example for shape = (2, 2) in row-major layout:
    /// yields: `0, 1, 2, 3`
    pub fn storage_indices(&self) -> StorageIndices {
        self.layout().storage_indices()
    }

    /// Returns an iterator over **dimension coordinates**.
    ///
    /// This iterator yields the multi-dimensional coordinates
    /// (e.g., `[i, j, k, ...]`) of each element in the array, independent
    /// of the physical storage layout.
    ///
    /// Example for shape = (2, 2):
    /// yields: `[0, 0], [0, 1], [1, 0], [1, 1]`
    pub fn dim_coordinates(&self) -> DimCoordinates {
        self.shape().dim_coordinates()
    }

    pub fn dims_coordinates<const N: usize>(&self) -> Result<DimNCoordinates<N>> {
        self.shape().dims_coordinates::<N>()
    }

    pub fn dim2_coordinates(&self) -> Result<DimNCoordinates<2>> {
        self.shape().dim2_coordinates()
    }

    pub fn dim3_coordinates(&self) -> Result<DimNCoordinates<3>> {
        self.shape().dim3_coordinates()
    }

    pub fn dim4_coordinates(&self) -> Result<DimNCoordinates<4>> {
        self.shape().dim4_coordinates()
    }

    pub fn dim5_coordinates(&self) -> Result<DimNCoordinates<5>> {
        self.shape().dim5_coordinates()
    }
}

impl<T: WithDType> NdArray<T> {
    pub fn matrix_view_unsafe(&self) -> Result<MatrixViewUsf<'_, T>> {
        MatrixViewUsf::from_ndarray(self)
    }

    pub fn vector_view_unsafe(&self) -> Result<VectorViewUsf<'_, T>> {
        VectorViewUsf::from_ndarray(self)
    }

    pub fn matrix_view<'a>(&'a self) -> Result<MatrixView<'a, T>> {
        MatrixView::from_ndarray(self)
    }

    pub fn matrix_view_mut<'a>(&'a mut self) -> Result<MatrixViewMut<'a, T>> {
        MatrixViewMut::from_ndarray_mut(self)
    }

    pub fn vector_view<'a>(&'a self) -> Result<VectorView<'a, T>> {
        VectorView::from_ndarray(self)
    }

    pub fn vector_view_mut<'a>(&'a mut self) -> Result<VectorViewMut<'a, T>> {
        VectorViewMut::from_ndarray_mut(self)
    }
}

impl<T: NumDType> NdArray<T> {
    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a.close(b, rtol, atol))
    }
}
