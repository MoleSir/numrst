use std::sync::{Arc, RwLock};
use rand_distr::{Distribution, StandardNormal, StandardUniform};
use crate::{Error, FloatDType, Layout, Result, Shape, Storage, WithDType};
use super::{NdArray, NdArrayId, NdArrayImpl};

impl<T: WithDType> NdArray<T> {
    pub fn new<A: ToNdArray<T>>(array: A) -> Result<Self> {
        let shape = array.shape()?;
        let storage = array.to_storage()?;
        Ok(Self::from_storage(storage, shape))
    }

    pub fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::zeros(&shape);
        Ok(Self::from_storage(storage, shape))
    }

    pub fn zero_like(&self) -> Result<Self> {
        Self::zeros(self.shape())
    }

    pub fn ones<S: Into<Shape>>(shape: S) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::ones(&shape);
        Ok(Self::from_storage(storage, shape))
    }

    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape())
    }

    pub fn rand<S: Into<Shape>>(min: T, max: T, shape: S) -> Result<Self> 
    where 
        StandardUniform: Distribution<T>
    {
        let shape = shape.into();
        let storage = Storage::rand_uniform(&shape, min, max)?;
        Ok(Self::from_storage(storage, shape))
    }

    pub fn rand_like(&self, min: T, max: T) -> Result<Self> 
    where 
        StandardUniform: Distribution<T>
    {
        Self::rand(min, max, self.shape())
    }

    pub fn fill<S: Into<Shape>>(shape: S, value: T) -> Result<Self> {
        let shape: Shape = shape.into();
        let storage = T::to_filled_storage(value, shape.element_count())?;
        Ok(Self::from_storage(storage, shape))
    }

    pub fn arange(start: T, end: T) -> Result<Self> {
        let storage = T::to_range_storage(start, end)?;
        let shape = storage.len();
        Ok(Self::from_storage(storage, shape))
    }

    pub fn from_vec<S: Into<Shape>>(vec: Vec<T>, shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        if shape.element_count() != vec.len() {
            return Err(Error::Msg(format!("shape' element_count {} != vec.len {}", shape.element_count(), vec.len())));
        }
        let storage = T::to_storage(vec)?;
        Ok(Self::from_storage(storage, shape))
    }

    pub(crate) fn from_storage<S: Into<Shape>>(storage: Storage<T>, shape: S) -> Self {
        let dtype = storage.dtype();
        let ndarray_ = NdArrayImpl {
            id: NdArrayId::new(),
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
            dtype
        };
        NdArray(Arc::new(ndarray_))
    }
}

impl<F: FloatDType> NdArray<F> 
where 
    StandardNormal: Distribution<F>
{
    pub fn randn<S: Into<Shape>>(mean: F, std: F, shape: S) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::rand_normal(&shape, mean, std)?;
        Ok(Self::from_storage(storage, shape))
    }

    pub fn randn_like(&self, mean: F, std: F) -> Result<Self> {
        Self::randn(mean, std, self.shape())
    }
}

pub trait ToNdArray<T> {
    fn shape(&self) -> Result<Shape>;
    fn to_storage(self) -> Result<Storage<T>>;
}

impl<D: WithDType> ToNdArray<D> for D {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::scalar())
    }

    fn to_storage(self) -> Result<Storage<D>> {
        D::to_storage([self].to_vec())
    }
}

impl<S: WithDType, const N: usize> ToNdArray<S> for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage<S>> {
        S::to_storage(self.to_vec())
    }
}

impl<S: WithDType> ToNdArray<S> for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage<S>> {
        S::to_storage(self.to_vec())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize> ToNdArray<S> 
    for &[[S; N2]; N1] 
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2)))
    }

    fn to_storage(self) -> Result<Storage<S>> {
        S::to_storage(self.concat())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> ToNdArray<S>
    for &[[[S; N3]; N2]; N1] 
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_storage(self) -> Result<Storage<S>> {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        S::to_storage(vec)
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> ToNdArray<S>
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_storage(self) -> Result<Storage<S>> {
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

impl<S: WithDType> ToNdArray<S> for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage<S>> {
        S::to_storage(self)
    }
}