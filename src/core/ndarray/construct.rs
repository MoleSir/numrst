use std::sync::{Arc, RwLock};
use rand_distr::{Distribution, StandardNormal, StandardUniform};
use crate::{Error, FloatDType, Layout, NumDType, Result, Shape, Storage, WithDType};
use super::{NdArray, NdArrayId, NdArrayImpl};

impl<T: WithDType> NdArray<T> {
    /// Creates a new `NdArray` from any supported Rust array or slice.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::new(&[1, 2, 3]).unwrap();
    /// println!("{}", a.shape());
    /// ```
    pub fn new<A: ToNdArray<T>>(array: A) -> Result<Self> {
        let shape = array.shape()?;
        let storage = array.to_storage()?;
        Ok(Self::from_storage(storage, shape))
    }

    /// Creates an array full with a constant `value`.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::full((2, 2), 7).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn full<S: Into<Shape>>(shape: S, value: T) -> Result<Self> {
        let shape: Shape = shape.into();
        let storage = Storage::new(vec![value; shape.element_count()]);
        Ok(Self::from_storage(storage, shape))
    }

    /// Creates a new `NdArray` directly from a storage buffer and shape.
    ///
    /// Typically used internally, but can also be used when you already
    /// have a `Storage<T>` prepared.
    pub(crate) fn from_storage<S: Into<Shape>>(storage: Storage<T>, shape: S) -> Self {
        let ndarray_ = NdArrayImpl {
            id: NdArrayId::new(),
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::contiguous(shape),
        };
        NdArray(Arc::new(ndarray_))
    }
}

impl<T: NumDType> NdArray<T> {
    /// Creates an array of zeros with the given shape.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::<f32>::zeros((2, 3)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::zeros(&shape);
        Ok(Self::from_storage(storage, shape))
    }

    /// Creates a zero-filled array with the same shape as `self`.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::<i32>::ones((2, 2)).unwrap();
    /// let b = a.zero_like().unwrap();
    /// println!("{}", b);
    /// ```
    pub fn zero_like(&self) -> Result<Self> {
        Self::zeros(self.shape())
    }

    /// Creates an array of ones with the given shape.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::<f64>::ones((3, 3)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::ones(&shape);
        Ok(Self::from_storage(storage, shape))
    }

    /// Creates a one-filled array with the same shape as `self`.
    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape())
    }

    /// Creates a 1-D array with values from `start` up to (but not including) `end`.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::arange(0., 5.).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn arange(start: T, end: T) -> Result<Self> {
        let storage = T::to_range_storage(start, end)?;
        let shape = storage.len();
        Ok(Self::from_storage(storage, shape))
    }

    /// Creates an array from a flat `Vec<T>` and explicit shape.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::from_vec(vec![1, 2, 3, 4], (2, 2)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn from_vec<S: Into<Shape>>(vec: Vec<T>, shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        if shape.element_count() != vec.len() {
            Err(Error::ElementSizeMismatch { expected: vec.len(), got: shape.element_count(), op: "from_vec" })?
        }
        let storage = Storage::new(vec);
        Ok(Self::from_storage(storage, shape))
    }
}

impl<T: WithDType + rand_distr::uniform::SampleUniform> NdArray<T> 
where 
    StandardUniform: Distribution<T>
{
    /// Creates an array with uniformly distributed random values in `[min, max)`.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::<f32>::rand(0., 1., (2, 3)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn rand<S: Into<Shape>>(min: T, max: T, shape: S) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::rand_uniform(&shape, min, max)?;
        Ok(Self::from_storage(storage, shape))
    }

    /// Creates a random array with the same shape as `self`.
    pub fn rand_like(&self, min: T, max: T) -> Result<Self> {
        Self::rand(min, max, self.shape())
    }
}

impl<F: FloatDType> NdArray<F> {
    /// Generate a 1-D `NdArray` of `num` evenly spaced values over the interval [start, stop).
    /// 
    /// # Example
    ///
    /// ```
    /// # use numrst::NdArray;
    /// let arr = NdArray::linspace(0.0, 1.0, 5).unwrap();
    /// assert_eq!(arr.to_vec(), [0.0, 0.2, 0.4, 0.6000000000000001, 0.8]);
    /// ```
    pub fn linspace(start: F, stop: F, num: usize) -> Result<Self> {
        let step = (stop - start) / F::from_usize(num);
        let vec: Vec<_> = std::iter::successors(Some(start), |&x| {
            let next = x + step;
            if next < stop { Some(next) } else { None }
        })
        .collect();

        let len = vec.len();
        let storage = Storage::new(vec);
        Ok(Self::from_storage(storage, len))
    }
}

impl<F: FloatDType> NdArray<F> 
where 
    StandardNormal: Distribution<F>
{
    /// Creates an array with normally distributed random values
    /// with given `mean` and `std`.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::<f64>::randn(0.0, 1.0, (2, 2)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn randn<S: Into<Shape>>(mean: F, std: F, shape: S) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::rand_normal(&shape, mean, std)?;
        Ok(Self::from_storage(storage, shape))
    }

    /// Creates a normal-distributed random array with the same shape as `self`.
    pub fn randn_like(&self, mean: F, std: F) -> Result<Self> {
        Self::randn(mean, std, self.shape())
    }
}

impl NdArray<bool> {
    /// Creates a boolean array filled with `true`.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::trues((2, 2)).unwrap();
    /// println!("{}", a);
    /// ```
    pub fn trues<S: Into<Shape>>(shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        let storage = Storage::new(vec![true; shape.element_count()]);
        Ok(Self::from_storage(storage, shape))
    }

    /// Creates a boolean array filled with `false`.
    pub fn falses<S: Into<Shape>>(shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        let storage = Storage::new(vec![false; shape.element_count()]);
        Ok(Self::from_storage(storage, shape))
    }
}

pub trait ToNdArray<T> {
    fn shape(&self) -> Result<Shape>;
    fn to_storage(self) -> Result<Storage<T>>;
}

impl<D: WithDType> ToNdArray<D> for D {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::scalar())
    }

    fn to_storage(self) -> Result<Storage<D>> {
        Ok(Storage::new([self].to_vec()))
    }
}

impl<S: WithDType, const N: usize> ToNdArray<S> for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage<S>> {
        Ok(Storage::new(self.to_vec()))
    }
}

impl<S: WithDType> ToNdArray<S> for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage<S>> {
        Ok(Storage::new(self.to_vec()))
    }
}

impl<S: WithDType, const N1: usize, const N2: usize> ToNdArray<S> 
    for &[[S; N2]; N1] 
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2)))
    }

    fn to_storage(self) -> Result<Storage<S>> {
        Ok(Storage::new(self.concat()))
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> ToNdArray<S>
    for &[[[S; N3]; N2]; N1] 
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_storage(self) -> Result<Storage<S>> {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        Ok(Storage::new(vec))

    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> ToNdArray<S>
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_storage(self) -> Result<Storage<S>> {
        let mut vec = Vec::with_capacity(N1 * N2 * N3 * N4);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                for i3 in 0..N3 {
                    vec.extend(self[i1][i2][i3])
                }
            }
        }
        Ok(Storage::new(vec))
    }
}

impl<S: WithDType> ToNdArray<S> for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))    
    }

    fn to_storage(self) -> Result<Storage<S>> {
        Ok(Storage::new(self))

    }
}