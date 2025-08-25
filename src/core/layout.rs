use crate::{Error, Result};
use super::{Dim, Shape};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Layout {
    shape: Shape,
    stride: Vec<usize>,
    start_offset: usize,
}

impl Layout {
    pub fn new<S: Into<Shape>>(shape: S, stride: Vec<usize>, start_offset: usize) -> Self {
        Self {
            shape: shape.into(), stride, start_offset
        }
    }
    
    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            start_offset: 0,
        }
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(&self.shape, "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn element_count(&self) -> usize {
        self.shape().element_count()
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dim >= dims.len() {
            Err(Error::DimOutOfRange { 
                shape: self.shape().clone(), 
                dim: dim as i32, 
                op: "narrow" 
            })?;
        }
        if start + len > dims[dim] {
            Err(Error::NarrowInvalidArgs {
                shape: self.shape.clone(),
                dim,
                start,
                len,
                msg: "start + len > dim_len",
            })?
        }

        let mut dims = dims.to_vec();
        dims[dim] = len;
        Ok(Self::new(
            dims, 
            self.stride.clone(),
            self.start_offset + self.stride[dim] * start
        ))
    }

    pub fn narrow_range(&self, dim: usize, start: usize, end: usize, step: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dim >= dims.len() {
            Err(Error::DimOutOfRange { 
                shape: self.shape().clone(), 
                dim: dim as i32, 
                op: "narrow" 
            })?;
        }
        
        let len = (start..end).step_by(step).len();
        if len == 0 {
            Err(Error::NarrowInvalidArgs {
                shape: self.shape.clone(),
                dim,
                start,
                len,
                msg: "len == 0",
            })?
        }
        if end > dims[dim] {
            Err(Error::NarrowInvalidArgs {
                shape: self.shape.clone(),
                dim,
                start,
                len,
                msg: "start + len > dim_len",
            })?
        }

        let mut dims = dims.to_vec();
        dims[dim] = len;
        let mut stride = self.stride.clone();
        stride[dim] *= step;

        Ok(Self::new(
            dims, 
            stride,
            self.start_offset + self.stride[dim] * start
        ))
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if rank <= dim1 || rank <= dim2 {
            Err(Error::UnexpectedNumberOfDims {
                expected: usize::max(dim1, dim2),
                got: rank,
                shape: self.shape().clone(),
            })?
        }

        let mut stride = self.stride().to_vec();
        let mut dims = self.shape().dims().to_vec();
        dims.swap(dim1, dim2);
        stride.swap(dim1, dim2);

        Ok(Self::new(dims, stride, self.start_offset))
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.rank() < self.shape().rank() {
            return Err(Error::BroadcastIncompatibleShapes {
                src_shape: self.shape().clone(),
                dst_shape: shape,
            });
        }

        let added_dims = shape.rank() - self.shape().rank();
        let mut stride = vec![0; added_dims];
        for (&dst_dim, (&src_dim, &src_stride)) in shape.dims()[added_dims..]
            .iter()
            .zip(self.dims().iter().zip(self.stride()))
        {
            let s = if dst_dim == src_dim {
                src_stride
            } else if src_dim != 1 {
                return Err(Error::BroadcastIncompatibleShapes {
                    src_shape: self.shape().clone(),
                    dst_shape: shape,
                });
            } else {
                0
            };
            stride.push(s)
        }
        Ok(Self {
            shape,
            stride,
            start_offset: self.start_offset,
        })
    }

    pub fn to_index(&self) -> LayoutIndex {
        LayoutIndex::from_layout(self)
    }
}

pub enum LayoutIndex<'a> {
    UnContiguousIndex(UnContiguousIndex<'a>),
    ContiguousIndex(ContiguousIndex),
}

impl<'a> LayoutIndex<'a> {
    pub fn from_layout(l: &'a Layout) -> Self {
        if l.is_contiguous() {
            Self::ContiguousIndex(ContiguousIndex::from_layout(l))
        } else {
            Self::UnContiguousIndex(UnContiguousIndex::from_layout(l))
        }
    }
}

impl<'a> Iterator for LayoutIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::ContiguousIndex(i) => i.next(),
            Self::UnContiguousIndex(i) => i.next(),
        }
    }
}

pub struct ContiguousIndex {
    storage_index: usize,
    end_index: usize, 
}

impl ContiguousIndex {
    fn from_layout(l: &Layout) -> Self {
        Self {
            storage_index: l.start_offset(),
            end_index: l.start_offset() + l.element_count(),
        }
    }
}

impl Iterator for ContiguousIndex {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.storage_index >= self.end_index {
            None
        } else {
            let index = self.storage_index;
            self.storage_index += 1;
            Some(index)
        }
    }
}

#[derive(Debug)]
pub struct UnContiguousIndex<'a> {
    next_storage_index: Option<usize>,
    multi_index: Vec<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
}

impl<'a> UnContiguousIndex<'a> {
    fn new(dims: &'a [usize], stride: &'a [usize], start_offset: usize) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_storage_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(start_offset)
        };
        UnContiguousIndex {
            next_storage_index,
            multi_index: vec![0; dims.len()],
            dims,
            stride,
        }
    }

    fn from_layout(l: &'a Layout) -> Self {
        Self::new(l.dims(), l.stride(), l.start_offset())
    }
}

impl Iterator for UnContiguousIndex<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = self.next_storage_index?;
        let mut updated = false;
        let mut next_storage_index = storage_index;
        for ((multi_i, max_i), stride_i) in self
            .multi_index
            .iter_mut()
            .zip(self.dims.iter())
            .zip(self.stride.iter())
            .rev()
        {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                next_storage_index += stride_i;
                break;
            } else {
                next_storage_index -= *multi_i * stride_i;
                *multi_i = 0
            }
        }
        self.next_storage_index = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use super::{Layout, LayoutIndex};

    #[test]
    fn test_strided_index1() {
        let layout = Layout::contiguous((2, 5, 4));
        let index = LayoutIndex::from_layout(&layout);
        for i in index {
            println!("{}", i);
        }
    }

    #[test]
    fn test_strided_index2() {
        let layout = Layout::contiguous((2, 3, 3));
        let layout = layout.narrow(1, 1, 1).unwrap();
        println!("{:?}", layout.stride());
        let index = LayoutIndex::from_layout(&layout);
        for i in index {
            println!("{}", i);
        }
    }
}

