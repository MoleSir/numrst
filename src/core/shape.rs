use std::vec;

use crate::{Error, Result};

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn scalar() -> Self {
        Self(vec![])
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn into_dims(self) -> Vec<usize> {
        self.0
    }

    pub fn dim(&self, dim: impl Dim) -> Result<usize> {
        let index = dim.to_index(self, "get dim")?;
        Ok(self.dims()[index])
    }

    pub fn element_count(&self) -> usize {
        self.dims().iter().product()
    }

    pub fn is_contiguous(&self, stride: &[usize]) -> bool {
        if self.rank() != stride.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.dims().iter()).rev() {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    pub fn extend(mut self, additional_dims: &[usize]) -> Self {
        self.0.extend(additional_dims);
        self
    }

    pub(crate) fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride = self.dims()
            .iter()
            .rev()
            .scan(1, |prod, u| {
                let prod_pre_mult = *prod;
                *prod *= u;
                Some(prod_pre_mult)
            })
            .collect::<Vec<_>>();
        stride.reverse();
        stride
    }

    pub fn dims0(&self) -> Result<()> {
        Shape::check_rank::<0>(self.dims())?;
        Ok(())
    } 

    pub fn dims1(&self) -> Result<(usize,)> {
        Shape::check_rank::<1>(self.dims())?;
        Ok((self.dims()[0],))
    } 

    pub fn dims2(&self) -> Result<(usize, usize)> {
        Shape::check_rank::<2>(self.dims())?;
        Ok((
            self.dims()[0],
            self.dims()[1],
        ))
    } 

    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        Shape::check_rank::<3>(self.dims())?;
        Ok((
            self.dims()[0],
            self.dims()[1],
            self.dims()[2],
        ))
    }
    
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        Shape::check_rank::<4>(self.dims())?;
        Ok((
            self.dims()[0],
            self.dims()[1],
            self.dims()[2],
            self.dims()[3],
        ))
    }
    
    pub fn dims5(&self) -> Result<(usize, usize, usize, usize, usize)> {
        Shape::check_rank::<5>(self.dims())?;
        Ok((
            self.dims()[0],
            self.dims()[1],
            self.dims()[2],
            self.dims()[3],
            self.dims()[4],
        ))
    }

    fn check_rank<const C: usize, >(dims: &[usize]) -> Result<()> {
        if dims.len() != C {
            return Err(Error::UnexpectedNumberOfDims {
                expected: C,
                got: dims.len(),
                shape: Shape::from(dims),
            });
        }
        Ok(())
    }
}

impl<const C: usize> From<&[usize; C]> for Shape {
    fn from(dims: &[usize; C]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}


impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self([d1].to_vec())
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.0)
    }
}

macro_rules! impl_from_tuple {
    ($tuple:ty, $($index:tt),+) => {
        impl From<$tuple> for Shape {
            fn from(d: $tuple) -> Self {
                Self([$(d.$index,)+].to_vec())
            }
        }
    };
}

impl_from_tuple!((usize,), 0);
impl_from_tuple!((usize, usize), 0, 1);
impl_from_tuple!((usize, usize, usize), 0, 1, 2);
impl_from_tuple!((usize, usize, usize, usize), 0, 1, 2, 3);
impl_from_tuple!((usize, usize, usize, usize, usize), 0, 1, 2, 3, 4);
impl_from_tuple!((usize, usize, usize, usize, usize, usize), 0, 1, 2, 3, 4, 5);

pub enum D {
    Minus1,
    Minus2,
    Minus(usize),
    Index(usize),
}

impl D {
    fn out_of_range(&self, shape: &Shape, op: &'static str) -> Error {
        let dim = match self {
            Self::Minus1 => -1,
            Self::Minus2 => -2,
            Self::Minus(u) => -(*u as i32),
            Self::Index(u) => *u as i32,
        };
        Error::DimOutOfRange {
            shape: shape.clone(),
            dim,
            op,
        }
    }
}

pub trait Dim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= shape.rank() {
            Err(Error::DimOutOfRange {
                shape: shape.clone(),
                dim: dim as i32,
                op,
            })
        } else {
            Ok(dim)
        }
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 if rank >= 1 => Ok(rank - 1),
            Self::Minus2 if rank >= 2 => Ok(rank - 2),
            Self::Minus(u) if *u > 0 && rank >= *u => Ok(rank - *u),
            Self::Index(u) => u.to_index(shape, op),
            _ => Err(self.out_of_range(shape, op)),
        }
    }
}

pub trait Dims {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>>;
    fn check_indexes(dims: &[usize], shape: &Shape, op: &'static str) -> Result<()> {
        for (i, &dim) in dims.iter().enumerate() {
            if dims[..i].contains(&dim) {
                return Err(Error::DuplicateDimIndex {
                    shape: shape.clone(),
                    dims: dims.to_vec(),
                    op,
                });
            }
            if dim >= shape.rank() {
                return Err(Error::DimOutOfRange {
                    shape: shape.clone(),
                    dim: dim as i32,
                    op,
                });
            }
        }
        Ok(())
    }
}

impl Dims for Vec<usize> {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Self::check_indexes(&self, shape, op)?;
        Ok(self)
    }
}

impl<const N: usize> Dims for [usize; N] {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Self::check_indexes(&self, shape, op)?;
        Ok(self.to_vec())
    }
}

impl Dims for &[usize] {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Self::check_indexes(&self, shape, op)?;
        Ok(self.to_vec())
    }
}

impl Dims for () {
    fn to_indexes(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(vec![])
    }
}

impl<D: Dim + Sized> Dims for D {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dim = self.to_index(shape, op)?;
        Ok([dim].to_vec())
    }
}

impl<D: Dim> Dims for (D,) {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dim = self.0.to_index(shape, op)?;
        Ok([dim].to_vec())
    }
}

impl<D1: Dim, D2: Dim> Dims for (D1, D2) {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        Ok([d0, d1].to_vec())
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> Dims for (D1, D2, D3) {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        Ok([d0, d1, d2].to_vec())
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim> Dims for (D1, D2, D3, D4) {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        Ok([d0, d1, d2, d3].to_vec())
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> Dims for (D1, D2, D3, D4, D5) {
    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        let d4 = self.4.to_index(shape, op)?;
        Ok([d0, d1, d2, d3, d4].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride() {
        let shape = Shape::from(());
        assert_eq!(shape.stride_contiguous(), Vec::<usize>::new());
        let shape = Shape::from(42);
        assert_eq!(shape.stride_contiguous(), [1]);
        let shape = Shape::from((42, 1337));
        assert_eq!(shape.stride_contiguous(), [1337, 1]);
        let shape = Shape::from((299, 792, 458));
        assert_eq!(shape.stride_contiguous(), [458 * 792, 458, 1]);
    }

    #[test]
    fn test_from_tuple() {
        let shape = Shape::from((2,));
        assert_eq!(shape.dims(), &[2]);
        let shape = Shape::from((2, 3));
        assert_eq!(shape.dims(), &[2, 3]);
        let shape = Shape::from((2, 3, 4));
        assert_eq!(shape.dims(), &[2, 3, 4]);
        let shape = Shape::from((2, 3, 4, 5));
        assert_eq!(shape.dims(), &[2, 3, 4, 5]);
        let shape = Shape::from((2, 3, 4, 5, 6));
        assert_eq!(shape.dims(), &[2, 3, 4, 5, 6]);
        let shape = Shape::from((2, 3, 4, 5, 6, 7));
        assert_eq!(shape.dims(), &[2, 3, 4, 5, 6, 7]);
    }
}
