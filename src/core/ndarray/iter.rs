use crate::{StorageIndices, StorageRef, WithDType};

use super::NdArray;

pub struct NdArrayIter<'a, T> {
    indexes: StorageIndices<'a>,
    storage: StorageRef<'a, T>,
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