use std::sync::{Arc, RwLock};
use crate::{LayoutIndex, Scalar, Storage};

use super::NdArray;

pub struct TensorIter<'a> {
    indexes: LayoutIndex<'a>,
    storage: Arc<RwLock<Storage>>
}

impl<'a> Iterator for TensorIter<'a> {
    type Item = Scalar;

    fn next(&mut self) -> Option<Scalar> {
        let index = self.indexes.next()?;
        return self.storage.read().unwrap().get(index)
    }
}

impl NdArray {
    pub fn iter(&self) -> TensorIter {
        TensorIter {
            indexes: self.0.layout.to_index(),
            storage: self.0.storage.clone()
        }
    }
}