use crate::{StorageIndices, WithDType};

use super::NdArray;

pub struct NdArrayIter<'a, T> {
    indexes: StorageIndices<'a>,
    storage: StorageRef<'a, T>,
}

pub struct StorageRef<'a, T>(std::sync::MappedRwLockReadGuard<'a, [T]>);
pub struct StorageMut<'a, T>(std::sync::MappedRwLockWriteGuard<'a, [T]>);

impl<'a, T: WithDType> StorageRef<'a, T> {
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.0.get(index).copied()
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.0[index]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, T: WithDType> StorageMut<'a, T> {
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        self.0.get(index).copied()
    }

    pub fn set(&mut self, index: usize, val: T) -> Option<()> {
        if index >= self.0.len() {
            None
        } else {
            self.set_unchecked(index, val);
            Some(())
        }
    }

    #[inline]
    pub fn get_unchecked(&self, index: usize) -> T {
        self.0[index]
    }

    #[inline]
    pub fn set_unchecked(&mut self, index: usize, val: T) {
        self.0[index] = val;
    }
}

impl<'a, T: WithDType> Iterator for NdArrayIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let index = self.indexes.next()?;
        return self.storage.get(index)
    }
}

impl<T: WithDType> NdArray<T> {
    pub fn iter(&self) -> NdArrayIter<T> {
        NdArrayIter {
            indexes: self.0.layout.storage_indices(),
            storage: self.storage_ref(0)
        }
    }

    pub fn storage_ref(&self, start_offset: usize) -> StorageRef<'_, T> {
        StorageRef(std::sync::RwLockReadGuard::map(self.0.storage.read().unwrap(), |s| &s.data()[start_offset..]))
    }

    pub fn storage_mut(&self, start_offset: usize) -> StorageMut<'_, T> {
        StorageMut(std::sync::RwLockWriteGuard::map(self.0.storage.write().unwrap(), |s| &mut s.data_mut()[start_offset..]))
    }
}

pub trait ResettableIterator: Iterator {
    fn reset(&mut self);
}

impl<'a, T: WithDType> ResettableIterator for NdArrayIter<'a, T> {
    fn reset(&mut self) {
        self.indexes.reset();
    }
}

impl<'a, T: WithDType> ExactSizeIterator for NdArrayIter<'a, T> {
    fn len(&self) -> usize {
        self.indexes.len()
    }
}