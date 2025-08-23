use rand::Rng;
use rand_distr::Distribution;
use crate::{Error, Result};
use super::{DType, Layout, Scalar, Shape};

pub enum Storage {
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Storage {
    pub fn zeros(shape: &Shape, dtype: DType) -> Self {
        match dtype {
            DType::I32 => Self::I32(vec![0; shape.element_count()]),
            DType::U32 => Self::U32(vec![0; shape.element_count()]),
            DType::F32 => Self::F32(vec![0.; shape.element_count()]),
            DType::F64 => Self::F64(vec![0.; shape.element_count()]),
        }
    }

    pub fn ones(shape: &Shape, dtype: DType) -> Self {
        match dtype {
            DType::I32 => Self::I32(vec![1; shape.element_count()]),
            DType::U32 => Self::U32(vec![1; shape.element_count()]),
            DType::F32 => Self::F32(vec![1.; shape.element_count()]),
            DType::F64 => Self::F64(vec![1.; shape.element_count()]),
        }
    }

    pub fn rand_uniform(shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<Self> {
        let elem_count = shape.element_count();
        match dtype {
            DType::U32 | DType::I32 => {
                Err(Error::UnsupportedDTypeForOp(dtype, "rand_uniform"))
            }
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform =
                    rand::distr::Uniform::new(min as f32, max as f32).map_err(|e| Error::Msg(e.to_string()))?;
                let mut rng = rand::rng();
                for _i in 0..elem_count {
                    data.push(rng.sample::<f32, _>(uniform))
                }
                Ok(Self::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand::distr::Uniform::new(min, max).map_err(|e| Error::Msg(e.to_string()))?;
                let mut rng = rand::rng();
                for _i in 0..elem_count {
                    data.push(rng.sample::<f64, _>(uniform))
                }
                Ok(Self::F64(data))
            }
        }
    }

    pub fn rand_normal(shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self> {
        let elem_count = shape.element_count();
        match dtype {
            DType::U32 | DType::I32 => {
                Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal"))
            }
            DType::F32 => {
                let mut data: Vec<f32> = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(mean as f32, std as f32)
                    .map_err(|e| Error::Msg(e.to_string()))?;
                let mut rng = rand::rng();
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(Self::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(mean, std)
                    .map_err(|e| Error::Msg(e.to_string()))?;
                let mut rng = rand::rng();
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(Self::F64(data))
            }
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::I32(_) => DType::I32,
            Self::U32(_) => DType::U32,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn get(&self, index: usize) -> Option<Scalar> {
        match self {
            Self::I32(vec) => vec.get(index).cloned().map(Scalar::I32),
            Self::U32(vec) => vec.get(index).cloned().map(Scalar::U32),
            Self::F32(vec) => vec.get(index).cloned().map(Scalar::F32),
            Self::F64(vec) => vec.get(index).cloned().map(Scalar::F64),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::I32(vec) => vec.len(),
            Self::U32(vec) => vec.len(),
            Self::F32(vec) => vec.len(),
            Self::F64(vec) => vec.len(),
        }
    }

    pub fn copy(&self, layout: &Layout) -> Result<Storage> {
        fn _copy<T: Copy>(vec: &[T], layout: &Layout) -> Vec<T> {
            let mut output = vec![];
            for index in layout.to_index() {
                output.push( vec[index] );
            }
            output
        }
        
        match self {
            Storage::U32(vec) => {
                let output = _copy(vec, layout);
                Ok(Storage::U32(output))
            }
            Storage::I32(vec) => {
                let output = _copy(vec, layout);
                Ok(Storage::I32(output))
            }
            Storage::F32(vec) => {
                let output = _copy(vec, layout);
                Ok(Storage::F32(output))
            }
            Storage::F64(vec) => {
                let output = _copy(vec, layout);
                Ok(Storage::F64(output))
            }
        }
    }
}