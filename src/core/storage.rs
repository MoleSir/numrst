use rand::rng;
use rand_distr::{Distribution, StandardNormal, StandardUniform, Uniform};
use crate::{Error, Result};
use super::{DType, FloatDType, Layout, NumDType, Shape, WithDType};

pub struct Storage<T>(Vec<T>);

impl<T: NumDType> Storage<T> {
    pub fn zeros(shape: &Shape) -> Self {
        Self(vec![T::zero(); shape.element_count()])
    }

    pub fn ones(shape: &Shape) -> Self {
        Self(vec![T::one(); shape.element_count()])

    }
}

impl<T: WithDType + rand_distr::uniform::SampleUniform> Storage<T> {
    pub fn rand_uniform(shape: &Shape, min: T, max: T) -> Result<Self> 
    where 
        StandardUniform: Distribution<T>,
    {
        let elem_count = shape.element_count();
        let mut rng = rng();
        let uniform = Uniform::new(min, max).map_err(|e| Error::Msg(e.to_string()))?;
        let v: Vec<T> = (0..elem_count)
            .map(|_| uniform.sample(&mut rng))
            .collect();

        Ok(Self(v))
    }
}

impl<F: FloatDType> Storage<F> {
    pub fn rand_normal(shape: &Shape, mean: F, std: F) -> Result<Self> 
    where 
        StandardNormal: Distribution<F>,
    {
        let elem_count = shape.element_count();
        let normal = rand_distr::Normal::new(mean, std).map_err(|e| Error::Msg(e.to_string()))?;
        let mut rng = rng();
        let v: Vec<F> = (0..elem_count)
            .map(|_| normal.sample(&mut rng))
            .collect();
        Ok(Self(v))
    }
}

impl<T: WithDType> Storage<T> {
    pub fn new<D: Into<Vec<T>>>(data: D) -> Self {
        Self(data.into())
    }

    pub fn data(&self) -> &[T] {
        &self.0
    }

    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.0.get(index).copied()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn copy(&self, layout: &Layout) -> Result<Self> {
        let mut output = vec![];
        for index in layout.to_index() {
            output.push( self.0[index] );
        }
        Ok(Self(output))
    }
}