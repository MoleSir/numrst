use std::sync::{Arc, RwLock};
use crate::{LayoutIndex, Storage, WithDType};

use super::NdArray;

pub struct NdArrayIter<'a, T> {
    indexes: LayoutIndex<'a>,
    storage: Arc<RwLock<Storage<T>>>
}

impl<'a, T: WithDType> Iterator for NdArrayIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let index = self.indexes.next()?;
        return self.storage.read().unwrap().get(index)
    }
}

impl<T: WithDType> NdArray<T> {
    pub fn iter(&self) -> NdArrayIter<T> {
        NdArrayIter {
            indexes: self.0.layout.to_index(),
            storage: self.0.storage.clone()
        }
    }
}