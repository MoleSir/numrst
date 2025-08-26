use std::fmt::Display;
use crate::{Result, WithDType};
use super::NdArray;

impl<T: WithDType> NdArray<T> {
    fn indexes(&self, indexers: &[Indexer]) -> Result<Self> {
        let mut x = self.clone();
        let mut current_dim = 0;
        for indexer in indexers.iter() {
            x = match indexer {
                Indexer::Select(n) => x.narrow(current_dim, *n, 1)?.squeeze(current_dim)?,
                Indexer::Range(range) => {
                    let out = x.narrow_range(current_dim, range)?;
                    current_dim += 1;
                    out
                }
            };
        }
        Ok(x)
    }
}

impl<T: WithDType> NdArray<T> {
    pub fn matrix_get(&self, row: usize, col: usize) -> Result<T> {
        self.index((row, col))?.to_scalar()
    }

    pub fn matrix_set(&self, row: usize, col: usize, val: T) -> Result<()> {
        self.index((row, col))?.set_scalar(val)
    }

    pub fn vector_get(&self, n: usize) -> Result<T> {
        self.index(n)?.to_scalar()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indexer {
    Select(usize),
    Range(Range),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Range {
    pub start: usize, 
    pub end: Option<usize>, 
    pub step: usize
}

impl Range {
    pub fn new(start: usize, end: Option<usize>, step: usize) -> Self {
        Self { start, end, step }
    }

    pub fn len(&self) -> usize {
        self.clone().count()
    }
}

impl Iterator for Range {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        match self.end {
            Some(end) => {
                if self.start < end {
                    let value = self.start;
                    self.start += self.step;
                    Some(value)
                } else {
                    None
                }
            }
            None => {
                let value = self.start;
                self.start += self.step;
                Some(value)
            }
        }
    }
}

impl Display for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let step_part = match self.step {
            1 => format!(""),
            _ => format!(":{}", self.step),
        };
        match self.end {
            Some(end) => write!(f, "{}:{}{}", self.start, end, step_part),
            None => write!(f, "{}:{}", self.start, step_part),
        }
    }
}

impl From<usize> for Indexer {
    fn from(index: usize) -> Self {
        Indexer::Select(index)
    }
}

impl From<Range> for Indexer {
    fn from(value: Range) -> Self {
        Indexer::Range(value)
    }
}

impl From<std::ops::Range<usize>> for Indexer {
    fn from(value: std::ops::Range<usize>) -> Self {
        let range = Range::new(value.start, Some(value.end), 1);
        range.into()
    }
}

impl From<std::ops::RangeFrom<usize>> for Indexer {
    fn from(value: std::ops::RangeFrom<usize>) -> Self {
        let range = Range::new(value.start, None, 1);
        range.into()
    }
}

impl From<std::ops::RangeFull> for Indexer {
    fn from(_: std::ops::RangeFull) -> Self {
        let range = Range::new(0, None, 1);
        range.into()
    }
}

pub trait IndexOp<T, D> {
    fn index(&self, index: T) -> Result<NdArray<D>>;
}

impl<I: Into<Indexer>, D: WithDType> IndexOp<I, D> for NdArray<D> {
    fn index(&self, index: I) -> Result<NdArray<D>> {
        self.indexes(&[index.into()])
    }
}

impl<I: Into<Indexer>, D: WithDType> IndexOp<(I,), D> for NdArray<D> {
    fn index(&self, (index,): (I,)) -> Result<NdArray<D>> {
        self.indexes(&[index.into()])
    }
}

macro_rules! index_op_tuple {
    ($($t:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($t),*, D: WithDType> IndexOp<($($t,)*), D> for NdArray<D>
        where
            $($t: Into<Indexer>,)*
        {
            fn index(&self, ($($t,)*): ($($t,)*)) -> Result<NdArray<D>> {
                self.indexes(&[$($t.into(),)*])
            }
        }
    };
}

index_op_tuple!(I1, I2);
index_op_tuple!(I1, I2, I3);
index_op_tuple!(I1, I2, I3, I4);
index_op_tuple!(I1, I2, I3, I4, I5);

#[macro_export]
macro_rules! rng {
    // rng!(start:end)
    ($start:tt : $end:tt) => {
        Range::new($start as usize, Some($end as usize), 1)
    };
    // rng!(start:end:step)
    ($start:tt : $end:tt : $step:tt) => {
        Range::new($start as usize, Some($end as usize), $step as usize)
    };
    // rng!(start:)
    ($start:tt :) => {
        Range::new($start as usize, None, 1)
    };
    // rng!(start::step)
    ($start:tt :: $step:tt) => {
        Range::new($start as usize, None, $step as usize)
    };
    // rng!(:$end)
    (: $end:tt) => {
        Range::new(0, Some($end as usize), 1)
    };
    // rng!(:$end:$step)
    (: $end:tt : $step:tt) => {
        Range::new(0, Some($end as usize), $step as usize)
    };
    // rng!(::$step)
    (:: $step:tt) => {
        Range::new(0, None, $step as usize)
    };
    // rng!(:)
    (:) => {
        Range::new(0, None, 1)
    };
}

#[cfg(test)]
#[allow(unused)]
mod test {
    use crate::DType;

    use super::*;

    #[test]
    fn test_index_scalar_dim_reduction() {
        let arr = NdArray::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();
        let sub = arr.index(1).unwrap();
        assert_eq!(sub.shape().dims(), &[5, 5]);

        let sub = arr.index((2, 3)).unwrap();
        assert_eq!(sub.shape().dims(), &[5]); 
    }

    #[test]
    fn test_index_range_basic() {
        let arr = NdArray::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();

        let sub = arr.index(rng!(1:3)).unwrap();
        assert_eq!(sub.shape().dims(), &[2, 5, 5]);

        let sub = arr.index((rng!(1:3), rng!(3:4), 1)).unwrap();
        assert_eq!(sub.shape().dims(), &[2, 1]);
    }

    #[test]
    fn test_index_full_and_mixed() {
        let arr = NdArray::<i32>::zeros((5, 5, 5)).unwrap();

        let sub = arr.index((rng!(1:3), .., 1..2)).unwrap();
        assert_eq!(sub.shape().dims(), &[2, 5, 1]);

        let sub = arr.index((2, .., rng!(0:2))).unwrap();
        assert_eq!(sub.shape().dims(), &[5, 2]);

        let sub = arr.index((rng!(0:2), rng!(2:5), rng!(1:3))).unwrap();
        assert_eq!(sub.shape().dims(), &[2, 3, 2]);
    }

    #[test]
    fn test_index_out_of_bounds() {
        let arr = NdArray::<i32>::zeros((5, 5, 5)).unwrap();
        let result = arr.index(10);
        assert!(result.is_err());

        let result = arr.index(rng!(3:10));
        assert!(result.is_err());
    }
    
    #[test]
    fn test_index_scalar_and_values() {
        let arr = NdArray::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();

        let sub = arr.index(1).unwrap();
        let expected = NdArray::arange(25, 50).unwrap().reshape((5, 5)).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));
    }

    #[test]
    fn test_index_range_values() {
        let arr = NdArray::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();

        let sub = arr.index(rng!(1:3)).unwrap();
        let expected = NdArray::arange(25, 75).unwrap().reshape((2, 5, 5)).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));
    }

    #[test]
    fn test_index_mixed_values() {
        let arr = NdArray::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();

        let sub = arr.index((2, 3)).unwrap();
        let expected = NdArray::arange(65, 70).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));

        let sub = arr.index((rng!(1:3), rng!(3:5), 2)).unwrap();
        let mut vals = Vec::new();
        for i in 1..3 {
            for j in 3..5 {
                vals.push(i * 25 + j * 5 + 2);
            }
        }
        let expected = NdArray::from_vec(vals, (2, 2)).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));
    }

    #[test]
    fn test_index_with_full_dim() {
        let arr = NdArray::arange(0, 125).unwrap().reshape((5, 5, 5)).unwrap();

        let sub = arr.index((rng!(1:3), .., 1..2)).unwrap();

        let expected = arr.index((rng!(1:3), rng!(0:5), rng!(1:2))).unwrap();
        assert!(sub.allclose(&expected, 0.0, 0.0));
    }

    #[test]
    fn test_macro() {
        let t = (0..12usize);
        let t = (2usize..);
        assert_eq!(rng!(1:10), Range {start:1, end: Some(10), step:1});

        assert!(
            rng!(1:20).zip((1..20))
                .all(|(a, b)| a == b)
        );
    
        assert!(
            rng!(1:13:3).zip((1..13).step_by(3))
                .all(|(a, b)| a == b)
        );
    
        assert!(
            rng!(1:).zip((1..).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(1::2).zip((1..).step_by(2).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(:20).zip((0..20usize))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(:20:5).zip((0..20usize).step_by(5))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(::2).zip((0..).step_by(2).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(:).zip((0..).take(100))
                .all(|(a, b)| a == b)
        );
    }
}

