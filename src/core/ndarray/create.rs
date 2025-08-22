use std::sync::{Arc, RwLock};

use crate::{DType, Layout, Result, Shape, Storage, WithDType};

use super::{NdArray, NdArrayId, NdArrayImpl};

impl NdArray {
    pub fn new<A: ToNdArray>(array: A) -> Result<Self> {
        let shape = array.shape()?;
        let storage = array.to_storage()?;
        let dtype = storage.dtype();
        let ndarray_impl = NdArrayImpl {
            id: NdArrayId::new(),
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
            dtype
        };
        Ok(Self(Arc::new(ndarray_impl)))
    }

    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::zeros(&shape, dtype);
        let ndarray_impl = NdArrayImpl {
            id: NdArrayId::new(),
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
            dtype
        };
        Ok(Self(Arc::new(ndarray_impl)))
    }

    pub fn zero_like(&self) -> Result<Self> {
        Self::zeros(self.shape(), self.dtype())
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::ones(&shape, dtype);
        let ndarray_impl = NdArrayImpl {
            id: NdArrayId::new(),
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
            dtype
        };
        Ok(Self(Arc::new(ndarray_impl)))
    }

    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape(), self.dtype())
    }

    pub fn rand<S: Into<Shape>>(min: f64, max: f64, shape: S, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::rand_uniform(&shape, dtype, min, max)?;
        let ndarray_impl = NdArrayImpl {
            id: NdArrayId::new(),
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
            dtype
        };
        Ok(Self(Arc::new(ndarray_impl)))
    }

    pub fn rand_like(&self, min: f64, max: f64) -> Result<Self> {
        Self::rand(min, max, self.shape(), self.dtype())
    }

    pub fn randn<S: Into<Shape>>(mean: f64, std: f64, shape: S, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::rand_normal(&shape, dtype, mean, std)?;
        let ndarray_impl = NdArrayImpl {
            id: NdArrayId::new(),
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
            dtype
        };
        Ok(Self(Arc::new(ndarray_impl)))
    }

    pub fn randn_like(&self, mean: f64, std: f64) -> Result<Self> {
        Self::randn(mean, std, self.shape(), self.dtype())
    }

    pub fn fill<S: Into<Shape>, D: WithDType>(shape: S, value: D) -> Result<Self> {
        let shape: Shape = shape.into();
        let storage = D::to_fill_storage(value, shape.element_count())?;
        let ndarray_impl = NdArrayImpl {
            id: NdArrayId::new(),
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
            dtype: value.dtype(),
        };
        Ok(Self(Arc::new(ndarray_impl)))
    }
}

pub trait ToNdArray {
    fn shape(&self) -> Result<Shape>;
    fn to_storage(self) -> Result<Storage>;
}

impl<D: WithDType> ToNdArray for D {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::scalar())
    }

    fn to_storage(self) -> Result<Storage> {
        D::to_storage([self].to_vec())
    }
}

impl<S: WithDType, const N: usize> ToNdArray for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage> {
        S::to_storage(self.to_vec())
    }
}

impl<S: WithDType> ToNdArray for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage> {
        S::to_storage(self.to_vec())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize> ToNdArray 
    for &[[S; N2]; N1] 
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2)))
    }

    fn to_storage(self) -> Result<Storage> {
        S::to_storage(self.concat())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> ToNdArray 
    for &[[[S; N3]; N2]; N1] 
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_storage(self) -> Result<Storage> {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        S::to_storage(vec)
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> ToNdArray
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_storage(self) -> Result<Storage> {
        let mut vec = Vec::with_capacity(N1 * N2 * N3 * N4);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                for i3 in 0..N3 {
                    vec.extend(self[i1][i2][i3])
                }
            }
        }
        S::to_storage(vec)
    }
}

impl<S: WithDType> ToNdArray for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage> {
        S::to_storage(self)
    }
}