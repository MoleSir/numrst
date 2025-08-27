use rand::rng;
use rand_distr::{Distribution, StandardNormal, StandardUniform, Uniform};
use crate::{Error, Result};
use super::{DType, FloatDType, Layout, NumDType, Shape, WithDType};

#[derive(Clone)]
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

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.0
    }

    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.0.get(index).copied()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn copy(&self, layout: &Layout) -> Self {
        let output: Vec<_> = layout.to_index()
            .map(|i| self.0[i])
            .collect();
        Self(output)
    }

    pub fn copy_map<F, U>(&self, layout: &Layout, f: F) -> Storage<U> 
    where 
        U: WithDType,
        F: Fn(T) -> U
    {
        let output: Vec<_> = layout.to_index()
            .map(|i| f(self.0[i]))
            .collect();
        Storage(output)
    }
}

