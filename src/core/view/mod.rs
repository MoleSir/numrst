mod matrix;
mod vector;
pub use matrix::*;
pub use vector::*;

use std::marker::PhantomData;
use crate::WithDType;

#[derive(Copy, Clone)]
pub struct StorageView<T>(pub(crate) *mut T);

#[derive(Copy, Clone)]
pub struct MatrixView<'a, T> {
    storage: StorageView<T>,
    shape: (usize, usize),
    strides: (usize, usize),
    _marker: PhantomData<&'a mut T>, 
} 

#[derive(Copy, Clone)]
pub struct VectorView<'a, T> {
    storage: StorageView<T>,
    len: usize,
    stride: usize,
    _marker: PhantomData<&'a mut T>, 
}

impl<'a, T: WithDType> StorageView<T> {
    #[inline]
    pub unsafe fn get(&self, storage_index: usize) -> T {
        let ptr = unsafe { self.0.add(storage_index) };
        let value = unsafe { *ptr };
        value
    }

    #[inline]
    pub unsafe fn get_ref(&self, storage_index: usize) -> &T {
        let ptr = unsafe { self.0.add(storage_index) };
        unsafe { &*ptr }
    }

    #[inline]
    pub unsafe fn get_mut(&mut self, storage_index: usize) -> &mut T {
        let ptr = unsafe { self.0.add(storage_index) };
        unsafe { &mut *ptr }
    }

    #[inline]
    pub unsafe fn set(&mut self, storage_index: usize, value: T) {
        let ptr = unsafe { self.0.add(storage_index) };
        unsafe { *ptr = value };
    }

    #[inline]
    pub unsafe fn add(&self, storage_index: usize) -> *mut T {
        unsafe { self.0.add(storage_index) }
    }
}
