use std::sync::{Arc, RwLock};
use rand::rng;
use rand_distr::{Distribution, StandardNormal, StandardUniform, Uniform};
use crate::{Error, Result};
use super::{DType, FloatDType, Layout, NumDType, Shape, WithDType};

#[derive(Clone)]
pub struct Storage<T>(Vec<T>);

impl<T: NumDType> Storage<T> {
    pub fn zeros(shape: &Shape) -> Self {
        Self(vec![T::zero(); shape.element_count()])
    }

    pub fn ones(shape: &Shape) -> Self {
        Self(vec![T::one(); shape.element_count()])
    }
}

impl<T: WithDType + rand_distr::uniform::SampleUniform> Storage<T> {
    pub fn rand_uniform(shape: &Shape, min: T, max: T) -> Result<Self> 
    where 
        StandardUniform: Distribution<T>,
    {
        let elem_count = shape.element_count();
        let mut rng = rng();
        let uniform = Uniform::new(min, max).map_err(|e| Error::Rand(e.to_string()))?;
        let v: Vec<T> = (0..elem_count)
            .map(|_| uniform.sample(&mut rng))
            .collect();

        Ok(Self(v))
    }
}

impl<F: FloatDType> Storage<F> {
    pub fn rand_normal(shape: &Shape, mean: F, std: F) -> Result<Self> 
    where 
        StandardNormal: Distribution<F>,
    {
        let elem_count = shape.element_count();
        let normal = rand_distr::Normal::new(mean, std).map_err(|e| Error::Rand(e.to_string()))?;
        let mut rng = rng();
        let v: Vec<F> = (0..elem_count)
            .map(|_| normal.sample(&mut rng))
            .collect();
        Ok(Self(v))
    }
}

impl<T: WithDType> Storage<T> {
    pub fn new<D: Into<Vec<T>>>(data: D) -> Self {
        Self(data.into())
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        &self.0
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.0
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    #[inline]
    pub fn copy_data(&self) -> Vec<T> {
        self.0.clone()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.0.get(index).copied()
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.0[index]
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: T) -> Option<()> {
        if index >= self.len() {
            None
        } else {
            self.0[index] = value;
            Some(())
        }
    }

    #[inline]
    pub fn set_unchecked(&mut self, index: usize, value: T) {
        self.0[index] = value;
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn copy(&self, layout: &Layout) -> Self {
        let output: Vec<_> = layout.storage_indices()
            .map(|i| self.0[i])
            .collect();
        Self(output)
    }

    pub fn copy_map<F, U>(&self, layout: &Layout, f: F) -> Storage<U> 
    where 
        U: WithDType,
        F: Fn(T) -> U
    {
        let output: Vec<_> = layout.storage_indices()
            .map(|i| f(self.0[i]))
            .collect();
        Storage(output)
    }
}

#[derive(Clone)]
pub struct StorageArc<T>(pub(crate) Arc<RwLock<Storage<T>>>);

impl<T: WithDType> StorageArc<T> {
    pub fn new(storage: Storage<T>) -> Self {
        Self(Arc::new(RwLock::new(storage)))
    }

    #[inline]
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, Storage<T>> {
        self.0.read().unwrap()
    }

    #[inline]
    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, Storage<T>> {
        self.0.write().unwrap()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.read().get(index)
    }

    #[inline]
    pub fn set(&mut self, index: usize, val: T) -> Option<()> {
        self.write().set(index, val)
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.read().get_unchecked(index)
    }

    #[inline]
    pub fn set_unchecked(&self, index: usize, val: T) {
        self.write().set_unchecked(index, val)
    }

    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }

    #[inline]
    pub fn get_ref(&self, start_offset: usize) -> StorageRef<'_, T> {
        StorageRef::Guard(std::sync::RwLockReadGuard::map(self.0.read().unwrap(), |s| &s.data()[start_offset..]))
    }

    #[inline]
    pub fn get_mut(&self, start_offset: usize) -> StorageMut<'_, T> {
        StorageMut::Guard(std::sync::RwLockWriteGuard::map(self.0.write().unwrap(), |s| &mut s.data_mut()[start_offset..]))
    }

    #[inline]
    pub fn get_ptr(&self, start_offset: usize) -> *mut T {
        let mut s = self.0.write().unwrap();
        let data = &mut s.data_mut()[start_offset..];
        data.as_mut_ptr()
    }
}

pub enum StorageRef<'a, T> {
    Guard(std::sync::MappedRwLockReadGuard<'a, [T]>),
    Slice(&'a [T]),
}

// pub struct StorageMut<'a, T>(std::sync::MappedRwLockWriteGuard<'a, [T]>);

pub enum StorageMut<'a, T> {
    Guard(std::sync::MappedRwLockWriteGuard<'a, [T]>),
    Slice(&'a mut[T]),
}

impl<'a, T: WithDType> StorageRef<'a, T> {
    pub fn clone(&'a self) -> Self {
        Self::Slice(&self.data())
    }

    pub fn slice(&'a self, index: usize) -> Self {
        Self::Slice(&self.data()[index..])
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.data().get(index).copied()
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.data()[index]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data().len()
    }

    pub fn data(&self) -> &[T] {
        match self {
            Self::Guard(gurad) => &gurad,
            Self::Slice(s) => s,
        }
    }
}

impl<'a, T: WithDType> StorageMut<'a, T> {
    pub fn clone(&'a self) -> StorageRef<'a, T> {
        StorageRef::Slice(self.data())
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.data().get(index).copied()
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.data()[index]
    }

    #[inline]
    pub fn set(&mut self, index: usize, val: T) -> Option<()> {
        if index >= self.len() {
            None
        } else {
            self.set_unchecked(index, val);
            Some(())
        }
    }

    #[inline]
    pub fn set_unchecked(&mut self, index: usize, val: T) {
        self.data_mut()[index] = val;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data().len()
    }

    pub fn data(&self) -> &[T] {
        match self {
            Self::Guard(gurad) => &gurad,
            Self::Slice(s) => s,
        }
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        match self {
            Self::Guard(gurad) => &mut gurad[0..],
            Self::Slice(s) => &mut s[0..],
        }
    }
}