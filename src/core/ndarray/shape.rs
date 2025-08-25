use std::sync::Arc;

use crate::{Dim, Error, Layout, Result, Shape, WithDType};

use super::{NdArray, NdArrayId, NdArrayImpl, Range};

impl<T: WithDType> NdArray<T> {
    /// Creates a new tensor with the specified dimension removed if its size was one.
    ///
    /// ```rust
    /// use numrst::{NdArray, DType, D};
    /// let a = NdArray::<f32>::zeros((2, 3, 1)).unwrap();
    ///
    /// let c = a.squeeze(2).unwrap();
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    /// ```
    pub fn squeeze<D: Dim>(&self, dim: D) -> Result<Self> {
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "squeeze")?;
        if dims[dim] == 1 {
            let mut dims = dims.to_vec();
            let mut strides = self.stride().to_vec();
            dims.remove(dim);
            strides.remove(dim);
            let ndarray_ = NdArrayImpl {
                id: NdArrayId::new(),
                storage: self.0.storage.clone(),
                layout: Layout::new(dims, strides, self.layout().start_offset()),
                dtype: self.dtype(),
            };
            Ok(Self(Arc::new(ndarray_)))
        } else {
            Err( Error::SqueezeDimNot1 { shape: self.shape().clone(), dim } )
        }
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    ///
    /// ```rust
    /// use numrst::{NdArray, DType, D};
    /// let a = NdArray::<f32>::zeros((2, 3)).unwrap();
    ///
    /// let c = a.unsqueeze(0).unwrap();
    /// assert_eq!(c.shape().dims(), &[1, 2, 3]);
    ///
    /// let c = a.unsqueeze(D::Minus1).unwrap();
    /// assert_eq!(c.shape().dims(), &[2, 3, 1]);
    /// ```
    pub fn unsqueeze<D: Dim>(&self, dim: D) -> Result<Self> {
        let mut dims = self.dims().to_vec();
        let mut strides = self.stride().to_vec();
        let dim = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        dims.insert(dim, 1);
        let stride = if dim < strides.len() { strides[dim] } else { 1 };
        strides.insert(dim, stride);
        let ndarray_ = NdArrayImpl {
            id: NdArrayId::new(),
            storage: self.0.storage.clone(),
            layout: Layout::new(dims, strides, self.layout().start_offset()),
            dtype: self.dtype(),
        };
        Ok(Self(Arc::new(ndarray_)))
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + len`.
    /// ```
    /// use numrst::NdArray;
    /// let a = NdArray::new(&[
    ///     [0f32, 1., 2.],
    ///     [3.  , 4., 5.],
    ///     [6.  , 7., 8.]
    /// ]).unwrap();
    ///
    /// let b = a.narrow(0, 1, 2).unwrap();
    /// assert_eq!(b.shape().dims(), &[2, 3]);
    ///
    /// let c = a.narrow(1, 1, 1).unwrap();
    /// assert_eq!(c.shape().dims(), &[3, 1]);
    /// ```
    pub fn narrow<D: Dim>(&self, dim: D, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "narrow")?;
        let err = |msg| {
            Err::<(), _>(Error::NarrowInvalidArgs {
                shape: self.shape().clone(),
                dim,
                start,
                len,
                msg,
            })
        };

        if start > dims[dim] {
            err("start > dim_len")?;
        }
        if start.saturating_add(len) > dims[dim] {
            err("start + len > dim_len")?
        }
        if start == 0 && dims[dim] == len {
            Ok(self.clone())
        } else {
            let layout = self.layout().narrow(dim, start, len)?;
            let ndarray_ = NdArrayImpl {
                id: NdArrayId::new(),
                storage: self.0.storage.clone(),
                layout,
                dtype: self.dtype(),
            };
            Ok(Self(Arc::new(ndarray_)))
        }
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start : end : step`.
    /// ```
    /// use numrst::{NdArray, DType, rng, Range};
    /// let a = NdArray::<i32>::zeros((5, 5, 5)).unwrap();
    ///
    /// let b = a.narrow(0, 1, 2).unwrap();
    /// assert_eq!(b.shape().dims(), &[2, 5, 5]);
    ///
    /// let c = a.narrow_range(1, &rng!(::2)).unwrap();
    /// assert_eq!(c.shape().dims(), &[5, 3, 5]);
    /// ```
    pub fn narrow_range<D: Dim>(&self, dim: D, range: &Range) -> Result<Self> {
        let dims = self.dims();
        let dim = dim.to_index(self.shape(), "narrow")?;
        let err = |msg| {
            Err::<(), _>(Error::NarrowRangeInvalidArgs {
                shape: self.shape().clone(),
                dim,
                range: range.clone(),
                msg,
            })
        };

        let end = match range.end {
            Some(end) => end,
            None => dims[dim],
        };
        if range.start > dims[dim] {
            err("start > dim_len")?;
        }
        if end > dims[dim] {
            err("end > dim_len")?
        }
        if range.start == 0 && dims[dim] == end && range.step == 1 {
            Ok(self.clone())
        } else {
            let layout = self.layout().narrow_range(dim, range.start, end, range.step)?;
            let ndarray_ = NdArrayImpl {
                id: NdArrayId::new(),
                storage: self.0.storage.clone(),
                layout,
                dtype: self.dtype(),
            };
            Ok(Self(Arc::new(ndarray_)))
        }
    }

    /// Reshape returns a tensor with the target shape provided that the number of elements of the
    /// original tensor is the same.
    /// If the input tensor is contiguous, this is a view on the original data. Otherwise this uses
    /// a new storage and copies the data over, the returned tensor is always contiguous.
    ///
    /// The shape can be specified using a tuple of `usize` and at most one `()` in which case
    /// the behavior is the same as when using `-1` in PyTorch: this dimension size is adjusted so
    /// as to match the number of elements in the tensor.
    /// 
    /// ```rust
    /// use numrst::{NdArray, DType, D};
    /// let a = NdArray::<f32>::zeros((2, 3)).unwrap();
    ///
    /// let c = a.reshape((1, 6)).unwrap();
    /// assert_eq!(c.shape().dims(), &[1, 6]);
    /// ```
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.element_count() != self.element_count() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: shape,
                op: "reshape",
            });
        }

        if self.is_contiguous() {
            let ndarray_ = NdArrayImpl {
                id: NdArrayId::new(),
                storage: self.0.storage.clone(),
                layout: Layout::contiguous_with_offset(shape, self.layout().start_offset()),
                dtype: self.dtype(),
            };
            Ok(NdArray(Arc::new(ndarray_)))
        } else {
            let storage = self.storage().copy(self.layout())?;
            Ok(Self::from_storage(storage, shape))
        }
    }
    
    /// Returns a ndarray that is a transposed version of the input, the given dimensions are
    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Self> {
        let dim1 = dim1.to_index(self.shape(), "transpose")?;
        let dim2 = dim2.to_index(self.shape(), "transpose")?;
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let tensor_ = NdArrayImpl {
            id: NdArrayId::new(),
            storage: self.0.storage.clone(),
            layout: self.layout().transpose(dim1, dim2)?,
            dtype: self.dtype(),
        };
        Ok(NdArray(Arc::new(tensor_)))
    }
}