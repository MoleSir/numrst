use std::vec;

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub(crate) Vec<usize>);

impl Shape {
    pub fn scalar() -> Self {
        Self(vec![])
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
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


    /// Check whether the two shapes are compatible for broadcast, and if it is the case return the
    /// broadcasted shape. This is to be used for binary pointwise ops.
    /// Copy from https://github.com/huggingface/candle
    pub fn broadcast_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<Shape> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = vec![0; bcast_ndims];
        for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
            let rev_idx = bcast_ndims - idx;
            let l_value = if lhs_ndims < rev_idx {
                1
            } else {
                lhs_dims[lhs_ndims - rev_idx]
            };
            let r_value = if rhs_ndims < rev_idx {
                1
            } else {
                rhs_dims[rhs_ndims - rev_idx]
            };
            *bcast_value = if l_value == r_value {
                l_value
            } else if l_value == 1 {
                r_value
            } else if r_value == 1 {
                l_value
            } else {
                Err(Error::ShapeMismatchBinaryOp {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    op,
                })?
            }
        }
        Ok(Shape::from(bcast_dims))
    }


    /// Returns an iterator over **dimension coordinates**.
    ///
    /// This iterator yields the multi-dimensional coordinates
    /// (e.g., `[i, j, k, ...]`) of each element in the array, independent
    /// of the physical storage layout.
    ///
    /// Example for shape = (2, 2):
    /// yields: `[0, 0], [0, 1], [1, 0], [1, 1]`
    pub fn dim_coordinates(&self) -> DimCoordinates {
        DimCoordinates::from_shape(self)
    }

    pub fn dims_coordinates<const N: usize>(&self) -> Result<DimNCoordinates<N>> {
        DimNCoordinates::<N>::from_shape(self)
    }

    pub fn dim2_coordinates(&self) -> Result<DimNCoordinates<2>> {
        DimNCoordinates::<2>::from_shape(self)
    }

    pub fn dim3_coordinates(&self) -> Result<DimNCoordinates<3>> {
        DimNCoordinates::<3>::from_shape(self)
    }

    pub fn dim4_coordinates(&self) -> Result<DimNCoordinates<4>> {
        DimNCoordinates::<4>::from_shape(self)
    }

    pub fn dim5_coordinates(&self) -> Result<DimNCoordinates<5>> {
        DimNCoordinates::<5>::from_shape(self)
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
}


//////////////////////////////////////////////////////////////////////////////////////
///                  DimCoordinates
//////////////////////////////////////////////////////////////////////////////////////

pub struct DimCoordinates {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl DimCoordinates {
    pub fn from_shape(shape: &Shape) -> Self {
        let rank = shape.rank();
        Self {
            shape: shape.dims().to_vec(),
            current: vec![0; rank],
            done: shape.is_scalar(),
        }
    }
}

impl Iterator for DimCoordinates {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        for i in (0..self.current.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                break; 
            } else {
                self.current[i] = 0;
                if i == 0 {
                    self.done = true;
                }
            }
        }

        Some(result)
    }
}

pub struct DimNCoordinates<const N: usize> {
    shape: [usize; N],
    current: [usize; N],
    done: bool,
}

impl<const N: usize> DimNCoordinates<N> {
    pub fn from_shape(from_shape: &Shape) -> Result<Self> {
        if from_shape.rank() == N {
            let mut shape = [0usize; N];
            for i in 0..N {
                shape[i] = from_shape.dims()[i];
            }

            let current = [0usize; N];
            
            Ok(Self {
                shape,
                current,
                done: N == 0
            })
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: N,
                got: from_shape.rank(),
                shape: Shape::from(from_shape.dims()),
            })
        }
    }
}

impl<const N: usize> Iterator for DimNCoordinates<N> {
    type Item = [usize; N];
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current;

        for i in (0..N).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                break; 
            } else {
                self.current[i] = 0;
                if i == 0 {
                    self.done = true;
                }
            }
        }

        Some(result)
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

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        if self.0.len() == 1 {
            write!(f, ",")?;
        }
        write!(f, ")")
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


macro_rules! extract_dims {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        pub fn $fn_name(dims: &[usize]) -> Result<$out_type> {
            if dims.len() != $cnt {
                Err(Error::UnexpectedNumberOfDims {
                    expected: $cnt,
                    got: dims.len(),
                    shape: Shape::from(dims),
                })
            } else {
                Ok($dims(dims))
            }
        }

        impl Shape {
            pub fn $fn_name(&self) -> Result<$out_type> {
                $fn_name(self.0.as_slice())
            }
        }

        impl<T: crate::WithDType> crate::NdArray<T> {
            pub fn $fn_name(&self) -> Result<$out_type> {
                self.shape().$fn_name()
            }
        }

        impl std::convert::TryInto<$out_type> for Shape {
            type Error = crate::Error;
            fn try_into(self) -> std::result::Result<$out_type, Self::Error> {
                self.$fn_name()
            }
        }
    };
}

extract_dims!(dims0, 0, |_: &[usize]| (), ());
extract_dims!(dims1, 1, |d: &[usize]| d[0], usize);
extract_dims!(dims2, 2, |d: &[usize]| (d[0], d[1]), (usize, usize));
extract_dims!(
    dims3,
    3,
    |d: &[usize]| (d[0], d[1], d[2]),
    (usize, usize, usize)
);
extract_dims!(
    dims4,
    4,
    |d: &[usize]| (d[0], d[1], d[2], d[3]),
    (usize, usize, usize, usize)
);
extract_dims!(
    dims5,
    5,
    |d: &[usize]| (d[0], d[1], d[2], d[3], d[4]),
    (usize, usize, usize, usize, usize)
);


pub trait Dim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize>;
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

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim > shape.rank() {
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

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 => Ok(rank),
            Self::Minus2 if rank >= 1 => Ok(rank - 1),
            Self::Minus(u) if *u > 0 && rank + 1 >= *u => Ok(rank + 1 - *u),
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

    #[test]
    fn test_dim_coordinates_2d() {
        let shape = Shape([2, 2].to_vec());
        let mut iter = shape.dim_coordinates();

        let expected = [
            [0, 0].to_vec(),
            [0, 1].to_vec(),
            [1, 0].to_vec(),
            [1, 1].to_vec(),
        ];

        for e in expected {
            let idx = iter.next();
            assert_eq!(idx.unwrap(), e);
        }

        // Iter should be exhausted
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dim_coordinates_2d_varied() {
        let shape = Shape([3, 1].to_vec());
        let mut iter = shape.dim_coordinates();

        let expected = [
            [0, 0].to_vec(),
            [1, 0].to_vec(),
            [2, 0].to_vec(),
        ];

        for e in expected {
            let idx = iter.next();
            assert_eq!(idx.unwrap(), e);
        }

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dim_coordinates_3d() {
        let shape = Shape([2, 2, 2].to_vec());
        let mut iter = shape.dim_coordinates();

        let mut collected = Vec::new();
        while let Some(idx) = iter.next() {
            collected.push(idx);
        }

        let expected = [
            [0, 0, 0].to_vec(),
            [0, 0, 1].to_vec(),
            [0, 1, 0].to_vec(),
            [0, 1, 1].to_vec(),
            [1, 0, 0].to_vec(),
            [1, 0, 1].to_vec(),
            [1, 1, 0].to_vec(),
            [1, 1, 1].to_vec(),
        ];

        assert_eq!(collected, expected);
    }

    #[test]
    fn test_dim_n_coordinates_2d() {
        let shape = Shape([2, 2].to_vec());
        let mut iter = shape.dim2_coordinates().unwrap();

        let expected = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        for e in expected {
            let idx = iter.next();
            assert_eq!(idx.unwrap(), e);
        }

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dim_n_coordinates_3d() {
        let shape = Shape([2, 2, 2].to_vec());
        let mut iter = shape.dim3_coordinates().unwrap();

        let expected = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ];

        for e in expected {
            let idx = iter.next();
            assert_eq!(idx.unwrap(), e);
        }

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dim_n_coordinates_wrong_dim() {
        let shape = Shape([2, 2].to_vec());

        // dim3_coordinates should return error
        assert!(shape.dim3_coordinates().is_err());
        assert!(shape.dims_coordinates::<3>().is_err());
    }

    #[test]
    fn test_dim_n_coordinates_empty_shape() {
        let shape = Shape(vec![]);
        let mut iter = shape.dims_coordinates::<0>().unwrap();
        let result = iter.next();
        assert_eq!(result, None);
    }
}
