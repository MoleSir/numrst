use crate::{linalg::LinalgError, Error, FloatDType, NdArray, NumDType, Range, Result, Storage, WithDType};
use std::marker::PhantomData;
use super::{StorageView, VectorView};

impl<'a, T: WithDType> VectorView<'a, T> {
    pub fn from_ndarray(array: &NdArray<T>) -> Result<Self> {
        let _ = array.dims1()?;

        Ok(Self {
            storage: StorageView(array.storage_ptr(array.layout().start_offset())),
            len: array.layout().dims()[0],
            stride: array.layout().stride()[0],
            _marker: PhantomData,
        })
    }

    pub unsafe fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            None
        } else {
            Some(unsafe { self.g(index) })
        }
    }

    pub unsafe fn g(&self, index: usize) -> T {
        let storage_index = self.storage_index(index);
        unsafe { self.storage.get(storage_index) }
    }

    pub unsafe fn set(&mut self, index: usize, value: T) -> Option<()> {
        if index >= self.len {
            None
        } else {
            unsafe { self.s(index, value); }
            Some(())
        }
    }

    pub unsafe fn s(&mut self, index: usize, value: T) {
        let storage_index = self.storage_index(index);
        let ptr = unsafe { self.storage.add(storage_index) };
        unsafe { *ptr = value };
    }

    pub unsafe fn copy(&self) -> NdArray<T> {
        let data: Vec<_> = (0..self.len)
            .map(|index| unsafe { self.g(index) })
            .collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, (self.len,))
    }

    pub fn take(&self, size: usize) -> Result<Self> {
        if size > self.len() {
            Err(Error::VectorIndexOutOfRange { len: self.len(), index: size })?;
        }
        Ok(Self {
            storage: self.storage.clone(),
            len: size,
            stride: self.stride,
            _marker: PhantomData,
        })
    }

    pub unsafe fn drop(&self, size: usize) -> Result<Self> {
        if size > self.len() {
            Err(Error::VectorIndexOutOfRange { len: self.len(), index: size })?;
        }
        Ok(Self {
            storage: StorageView( unsafe { self.storage.add(self.storage_index(size)) } ),
            len: self.len - size,
            stride: self.stride,
            _marker: PhantomData,
        })
    }

    pub unsafe fn slice<R: Into<Range>>(&self, range: R) -> Result<Self> {
        let range = range.into();
        let end = range.end.unwrap_or(self.len);
        let end = end.min(self.len);
        let start = range.start;
        let step = range.step;

        let stride = self.stride * step;
        let storage = StorageView( unsafe { self.storage.add(self.storage_index(start)) } );
        let len = (start..end).step_by(2).count();
        
        Ok(Self {
            storage,
            len,
            stride,
            _marker: PhantomData,
        })
    }

    pub fn swap(&mut self, other: &mut Self) -> Result<()> {
        if self.len() != other.len() {
            return Err(LinalgError::VectorLenMismatch { len1: self.len, len2: other.len, op: "swap" })?;
        }

        let stride = self.stride;
        let storage_index = |index: usize| {
            stride * index
        };

        let len = self.len();
        (0..len).into_iter()
            .map(|index| (storage_index(index), storage_index(index)))
            .for_each(|(self_index, other_index)| {
                unsafe {
                    let self_value = self.storage.get(self_index);
                    let other_value = other.storage.get(other_index);
                    self.storage.set(self_index, other_value);
                    other.storage.set(other_index, self_value);
                }
            });

        Ok(())
    }

    pub unsafe fn eqal(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            false 
        } else {
            unsafe {
                self.iter().zip(other.iter()).all(|(a, b)| a == b)
            }
        }
    }

    pub unsafe fn iter(&'a self) -> VectorIterUsf<'a, T> {
        VectorIterUsf { 
            view: self.clone(), 
            index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn storage_index(&self, index: usize) -> usize {
        self.stride * index
    }
}

impl<'a, T: FloatDType> VectorView<'a, T> {
    pub unsafe fn norm(&self) -> T {
        let v = unsafe { self.iter().map(|v| v.powi(2)).sum::<T>() };
        v.sqrt()
    }
}

impl<'a, T: NumDType> VectorView<'a, T> {
    pub unsafe fn dot(&self, rhs: &Self) -> Result<T> {
        if self.len() != rhs.len() {
            Err(Error::LenMismatchVectorDot { lhs: self.len(), rhs: rhs.len() })?;
        }
        unsafe { Ok(self.iter().zip(rhs.iter())
            .map(|(a, b)| a * b)
            .sum::<T>()) }
    }

    pub unsafe fn mul_assign(&'a mut self, mul: T) {
        for i in 0..self.len {
            let v = unsafe { self.g(i) };
            unsafe { self.s(i, v * mul); }
        }
    }
}

pub struct VectorIterUsf<'a, T: WithDType> {
    view: VectorView<'a, T>,
    index: usize,
}

impl<'a, T: WithDType> Iterator for VectorIterUsf<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.len {
            None
        } else {
            let value = unsafe { self.view.g(self.index) };
            self.index += 1;
            Some(value)
        }
    }
}

impl<'a, T: WithDType> std::ops::Index<usize> for VectorView<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        let storage_index = self.storage_index(index);
        unsafe { self.storage.get_ref(storage_index) }
    }
}

impl<'a, T: WithDType> std::ops::IndexMut<usize> for VectorView<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        let storage_index = self.storage_index(index);
        unsafe { self.storage.get_mut(storage_index) }
    }
}
