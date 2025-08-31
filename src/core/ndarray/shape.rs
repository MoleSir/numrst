use std::sync::Arc;

use crate::{Dim, Error, Layout, Result, Shape, Storage, WithDType};

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
            let mut strides = self.layout().stride().to_vec();
            dims.remove(dim);
            strides.remove(dim);
            let ndarray_ = NdArrayImpl {
                id: NdArrayId::new(),
                storage: self.0.storage.clone(),
                layout: Layout::new(dims, strides, self.layout().start_offset()),
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
        let mut strides = self.layout().stride().to_vec();
        let dim = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        dims.insert(dim, 1);
        let stride = if dim < strides.len() { strides[dim] } else { 1 };
        strides.insert(dim, stride);
        let ndarray_ = NdArrayImpl {
            id: NdArrayId::new(),
            storage: self.0.storage.clone(),
            layout: Layout::new(dims, strides, self.layout().start_offset()),
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
            };
            Ok(Self(Arc::new(ndarray_)))
        }
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start : end : step`.
    /// 
    /// TODO: set `range` to a trait param
    /// 
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
            };
            Ok(NdArray(Arc::new(ndarray_)))
        } else {
            let storage = self.storage().copy(self.layout());
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
        };
        Ok(NdArray(Arc::new(tensor_)))
    }

    /// Concatenates two or more tensors along a particular dimension.
    ///
    /// All tensors must of the same rank, and the output will have
    /// the same rank
    ///
    /// ```rust
    /// use numrst::NdArray;
    /// let a = NdArray::<f32>::zeros((2, 3)).unwrap();
    /// let b = NdArray::<f32>::zeros((2, 3)).unwrap();
    ///
    /// let c = NdArray::cat(&[&a, &b], 0).unwrap();
    /// assert_eq!(c.dims(), &[4, 3]);
    ///
    /// let c = NdArray::cat(&[&a, &b], 1).unwrap();
    /// assert_eq!(c.dims(), &[2, 6]);
    /// ```
    pub fn cat<A: AsRef<NdArray<T>>, D: Dim>(arrs: &[A], dim: D) -> Result<Self> {
        // check shape
        if arrs.is_empty() {
            Err(Error::OpRequiresAtLeastOneNdArray { op: "cat" })?
        }
    
        // first arr's infomation
        let arr0 = &arrs[0];
        let rank0 = arr0.as_ref().rank();

        // cat_dim must be valid!
        let cat_dim = dim.to_index(arr0.as_ref().shape(), "cat")?;
        let mut target_dims = arr0.as_ref().dims().to_vec();
        target_dims[cat_dim] = 0;
        let mut dim_offsets = vec![];

        for (_arr_index, arr) in arrs.iter().enumerate() {
            // check shape 
            let rank = arr.as_ref().rank();
            if rank != rank0 {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arr.as_ref().rank(),
                    shape: arr.as_ref().shape().clone(),
                })?
            }

            // zip arr0's dims and arr's dims
            for (dim_index, (v1, v2)) in arr0.as_ref().dims().iter()
                                                                    .zip(arr.as_ref().dims().iter())
                                                                    .enumerate()
            {
                // accumalte the cat dim
                if dim_index == cat_dim {
                    dim_offsets.push(target_dims[cat_dim]);
                    target_dims[cat_dim] += v2;
                }

                // all other dims should be same
                if dim_index != cat_dim && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_index,
                        first_shape: arr0.as_ref().shape().clone(),
                        n: dim_index + 1,
                        nth_shape: arr0.as_ref().shape().clone(),
                    })?
                }
            }
        }
        
        // Now, all arr in arrs has same rank, and except `cat_dim`, all dims are equal
        // [ (a, n1, b, c), (a, n2, b, c), ... , (a, nk, b, c).... ]
        // target_dims = (a, n1+n2+...+nk, b, c)

        let target_shape: Shape = target_dims.into();
        
        // Create a new storgae and copy
        let mut dst: Vec<T> = Vec::with_capacity(target_shape.element_count());
        unsafe { dst.set_len(target_shape.element_count()) };
        let res_arr = Self::from_storage(Storage::new(dst), target_shape);

        for (arr_index, arr) in arrs.iter().enumerate() {
            // Take sub ndarray 
            let sub_res_arr = res_arr.narrow(cat_dim, dim_offsets[arr_index], arr.as_ref().dims()[cat_dim])?;
            assert_eq!(sub_res_arr.shape(), arr.as_ref().shape());
            sub_res_arr.copy_from(arr.as_ref())?;
        }

        Ok(res_arr)
    }

    /// Stacks two or more tensors along a particular dimension.
    ///
    /// All tensors must have the same rank, and the output has one additional rank
    ///
    /// ```rust
    /// use numrst::NdArray;
    /// let a = NdArray::<f32>::zeros((2, 3)).unwrap();
    /// let b = NdArray::<f32>::zeros((2, 3)).unwrap();
    ///
    /// let c = NdArray::stack(&[&a, &b], 0).unwrap();
    /// assert_eq!(c.dims(), &[2, 2, 3]);
    ///
    /// let c = NdArray::stack(&[&a, &b], 2).unwrap();
    /// assert_eq!(c.dims(), &[2, 3, 2]);
    /// ```
    pub fn stack<A: AsRef<NdArray<T>>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneNdArray { op: "stack" })?
        }
        let dim = dim.to_index_plus_one(args[0].as_ref().shape(), "stack")?;
        let args = args
            .iter()
            .map(|t| t.as_ref().unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        Self::cat(&args, dim)
    }

    /// Splits a tensor along a specified dimension into multiple sub-tensors.
    ///
    /// The tensor is split along the given `dim` into as many sub-tensors as
    /// the size of that dimension. Each sub-tensor has the same shape as the
    /// original tensor, except the size along `dim` becomes 1.
    ///
    /// ```rust
    /// use numrst::NdArray;
    ///
    /// let a = NdArray::new(&[[1, 2], [3, 4], [5, 6]]).unwrap();
    ///
    /// // Split along axis 0 (rows)
    /// let splits = a.split(0).unwrap();
    /// assert_eq!(splits.len(), 3);
    /// assert_eq!(splits[0].to_vec(), [1, 2]);
    /// assert_eq!(splits[1].to_vec(), [3, 4]);
    /// assert_eq!(splits[2].to_vec(), [5, 6]);
    ///
    /// // Split along axis 1 (columns)
    /// let splits = a.split(1).unwrap();
    /// assert_eq!(splits.len(), 2);
    /// assert_eq!(splits[0].to_vec(), [1, 3, 5]);
    /// assert_eq!(splits[1].to_vec(), [2, 4, 6]);
    ///
    /// // 1D array
    /// let b = NdArray::new(&[10, 20, 30]).unwrap();
    /// let splits = b.split(0).unwrap();
    /// assert_eq!(splits.len(), 3);
    /// assert_eq!(splits[0].to_vec(), [10]);
    /// assert_eq!(splits[1].to_vec(), [20]);
    /// assert_eq!(splits[2].to_vec(), [30]);
    /// ```
    pub fn split<D: Dim>(&self, dim: D) -> Result<Vec<Self>> {
        let split_index = dim.to_index(self.shape(), "split")?;
        let split_dim_size = self.dims()[split_index];
        let mut splited_shape = self.dims().to_vec();
        splited_shape.remove(split_index);
        let splited_shape: Shape = splited_shape.into();
        

        let mut vec = vec![];
        for i in 0..split_dim_size {
            let mut data: Vec<T> = Vec::with_capacity(splited_shape.element_count());
            unsafe { data.set_len(splited_shape.element_count()) };
            let storage = Storage::new(data);
            let arr = Self::from_storage(storage, splited_shape.clone());
            
            // Copy data
            let sub_self = self.narrow(split_index, i, 1)?.squeeze(split_index)?;
            assert_eq!(sub_self.dims(), splited_shape.dims());
            arr.assign(sub_self)?;

            vec.push(arr);
        }   

        Ok(vec)
    }

    /// Flattens the input tensor on the dimension indexes from `start_dim` to `end_dim` (both
    /// inclusive).
    pub fn flatten<D1: Dim, D2: Dim>(&self, start_dim: D1, end_dim: D2) -> Result<Self> {
        self.flatten_(Some(start_dim), Some(end_dim))
    }

    /// Flattens the input tensor on the dimension indexes from `0` to `end_dim` (inclusive).
    pub fn flatten_to<D: Dim>(&self, end_dim: D) -> Result<Self> {
        self.flatten_(None::<usize>, Some(end_dim))
    }

    /// Flattens the input tensor on the dimension indexes from `start_dim` (inclusive) to the last
    /// dimension.
    pub fn flatten_from<D: Dim>(&self, start_dim: D) -> Result<Self> {
        self.flatten_(Some(start_dim), None::<usize>)
    }

    /// Flattens the input tensor by reshaping it into a one dimension tensor.
    /// 
    /// ```rust
    /// use numrst::NdArray;
    /// let arr = NdArray::new(&[[0f32, 1.], [2., 3.], [4., 5.]]).unwrap();
    /// let arr = arr.flatten_all().unwrap();
    /// assert_eq!(arr.to_vec(), [0., 1., 2., 3., 4., 5.]);
    /// ```
    pub fn flatten_all(&self) -> Result<Self> {
        self.flatten_(None::<usize>, None::<usize>)
    }

    /// Repeat this tensor along the specified dimensions.
    pub fn repeat<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let repeats: Shape = shape.into();
        let mut repeats = repeats.dims().to_vec();

        if repeats.len() > self.rank() {
            Err(Error::RepeatRankOutOfRange { repeats: repeats.clone().into(), shape: self.shape().into() })?;
        } else if repeats.len() > self.rank() {
            for _ in 0..(repeats.len() - self.rank()) {
                repeats.push(1);
            }
        }

        let mut arr = self.clone();

        for (idx, &repeat) in repeats.iter().enumerate() {
            if repeat > 1 {
                arr = NdArray::cat(&vec![&arr; repeat], idx)?
            }
        }
        Ok(arr)
    }

    /// Repeat this tensor along the specified dimension with specified times
    pub fn repeat_dim<D: Dim>(&self, dim: D, times: usize) -> Result<Self> {
        if times == 0 {
            self.squeeze(dim)
        } else if times == 1 {
            Ok(self.clone())
        } else {
            NdArray::cat(&vec![self; times], dim)
        }
    }

    fn flatten_<D1: Dim, D2: Dim>(
        &self,
        start_dim: Option<D1>,
        end_dim: Option<D2>,
    ) -> Result<Self> {
        if self.rank() == 0 {
            self.reshape(1)
        } else {
            let start_dim = match start_dim {
                None => 0,
                Some(dim) => dim.to_index(self.shape(), "flatten")?,
            };
            let end_dim = match end_dim {
                None => self.rank() - 1,
                Some(dim) => dim.to_index(self.shape(), "flatten")?,
            };
            if start_dim < end_dim {
                let dims = self.dims();
                let mut dst_dims = dims[..start_dim].to_vec();
                dst_dims.push(dims[start_dim..end_dim + 1].iter().product::<usize>());
                if end_dim + 1 < dims.len() {
                    dst_dims.extend(&dims[end_dim + 1..]);
                }
                self.reshape(dst_dims)
            } else {
                Ok(self.clone())
            }
        }
    }
}

impl<T: WithDType> AsRef<NdArray<T>> for NdArray<T> {
    fn as_ref(&self) -> &NdArray<T> {
        self
    }
}

#[cfg(test)]
#[allow(unused)]
mod test {
    use super::*;

    #[test]
    fn test_cat_1d() -> Result<()> {
        let a = NdArray::new(&[1, 2, 3])?;
        let b = NdArray::new(&[4, 5, 6])?;
    
        let c = NdArray::cat(&[a, b], 0)?;
        assert_eq!(c.dims(), [6]);
        assert_eq!(c.to_vec(), [1, 2, 3, 4, 5, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_cat_2d_axis0() -> Result<()> {
        let a = NdArray::new(&[[1, 2], [3, 4]])?;
        let b = NdArray::new(&[[5, 6]])?;
    
        let c = NdArray::cat(&[a, b], 0)?;
        assert_eq!(c.dims(), [3, 2]);
        assert_eq!(c.to_vec(), [1, 2, 3, 4, 5, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_cat_2d_axis1() -> Result<()> {
        let a = NdArray::new(&[[1, 2], [3, 4]])?;
        let b = NdArray::new(&[[5], [6]])?;
    
        let c = NdArray::cat(&[a, b], 1)?;
        assert_eq!(c.dims(), [2, 3]);
        assert_eq!(c.to_vec(), [1, 2, 5, 3, 4, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_cat_3d() -> Result<()> {
        let a = NdArray::full((2, 2, 2), 1)?;
        let b = NdArray::full((2, 2, 2), 2)?;
    
        let c = NdArray::cat(&[a, b], 0)?;
        assert_eq!(c.dims(), [4, 2, 2]);
    
        let c2 = NdArray::cat(&[c.clone(), c.clone()], 1)?;
        assert_eq!(c2.dims(), [4, 4, 2]);
    
        Ok(())
    }
    
    #[test]
    fn test_cat_shape_mismatch() {
        let a = NdArray::new(&[1, 2, 3]).unwrap();
        let b = NdArray::new(&[[1, 2], [3, 4]]).unwrap();
    
        let res = NdArray::cat(&[a, b], 0);
        assert!(res.is_err());
    }
    
    #[test]
    fn test_cat_bool() -> Result<()> {
        let a = NdArray::new(&[[true, false]])?;
        let b = NdArray::new(&[[false, true]])?;
    
        let c = NdArray::cat(&[a, b], 0)?;
        assert_eq!(c.dims(), [2, 2]);
        assert_eq!(c.to_vec(), [true, false, false, true]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_1d_axis0() -> Result<()> {
        let a = NdArray::new(&[1, 2, 3])?;
        let b = NdArray::new(&[4, 5, 6])?;
    
        let c = NdArray::stack(&[a, b], 0)?;
        assert_eq!(c.dims(), [2, 3]); 
        assert_eq!(c.to_vec(), [1, 2, 3, 4, 5, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_1d_axis1() -> Result<()> {
        let a = NdArray::new(&[1, 2, 3])?;
        let b = NdArray::new(&[4, 5, 6])?;
    
        let c = NdArray::stack(&[a, b], 1)?;
        assert_eq!(c.dims(), [3, 2]);
        assert_eq!(c.to_vec(), [1, 4, 2, 5, 3, 6]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_2d_axis0() -> Result<()> {
        let a = NdArray::new(&[[1, 2], [3, 4]])?;
        let b = NdArray::new(&[[5, 6], [7, 8]])?;
    
        let c = NdArray::stack(&[a, b], 0)?;
        assert_eq!(c.dims(), [2, 2, 2]);
        assert_eq!(c.to_vec(), [1, 2, 3, 4, 5, 6, 7, 8]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_2d_axis1() -> Result<()> {
        let a = NdArray::new(&[[1, 2], [3, 4]])?;
        let b = NdArray::new(&[[5, 6], [7, 8]])?;
    
        let c = NdArray::stack(&[a, b], 1)?;
        assert_eq!(c.dims(), [2, 2, 2]);
        assert_eq!(c.to_vec(), [1, 2, 5, 6, 3, 4, 7, 8]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_2d_axis2() -> Result<()> {
        let a = NdArray::new(&[[1, 2], [3, 4]])?;
        let b = NdArray::new(&[[5, 6], [7, 8]])?;
    
        let c = NdArray::stack(&[a, b], 2)?;
        assert_eq!(c.dims(), [2, 2, 2]);
        assert_eq!(c.to_vec(), [1, 5, 2, 6, 3, 7, 4, 8]);
    
        Ok(())
    }
    
    #[test]
    fn test_stack_shape_mismatch() {
        let a = NdArray::new(&[1, 2, 3]).unwrap();
        let b = NdArray::new(&[4, 5]).unwrap();
    
        let res = NdArray::stack(&[a, b], 0);
        assert!(res.is_err());
    }
    
    #[test]
    fn test_split_1d() -> Result<()> {
        let a = NdArray::new(&[10, 20, 30, 40])?;
        let splits = a.split(0)?; // axis 0
    
        assert_eq!(splits.len(), 4);
        assert_eq!(splits[0].to_vec(), [10]);
        assert_eq!(splits[1].to_vec(), [20]);
        assert_eq!(splits[2].to_vec(), [30]);
        assert_eq!(splits[3].to_vec(), [40]);
    
        Ok(())
    }

    #[test]
    fn test_split_2d_axis0() -> Result<()> {
        let a = NdArray::new(&[[1, 2], [3, 4], [5, 6], [7, 8]])?;
        let splits = a.split(0)?;
        
        assert_eq!(splits.len(), 4);
        assert_eq!(splits[0].to_vec(), [1, 2]);
        assert_eq!(splits[1].to_vec(), [3, 4]);
        assert_eq!(splits[2].to_vec(), [5, 6]);
        assert_eq!(splits[3].to_vec(), [7, 8]);
        
        Ok(())
    }
    
    #[test]
    fn test_split_2d_axis1() -> Result<()> {
        let a = NdArray::new(&[[1, 2, 3], [4, 5, 6]])?;
        let splits = a.split(1)?;
        
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].to_vec(), [1, 4]); 
        assert_eq!(splits[1].to_vec(), [2, 5]); 
        assert_eq!(splits[2].to_vec(), [3, 6]); 
        
        Ok(())
    }
    
    #[test]
    fn test_split_3d_axis2() -> Result<()> {
        let a = NdArray::new(&[
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ])?;
        let splits = a.split(2)?;
        
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].to_vec(), [1, 3, 5, 7]); 
        assert_eq!(splits[1].to_vec(), [2, 4, 6, 8]); 
        
        Ok(())
    }
    
    #[test]
    fn test_split_single_element() -> Result<()> {
        let a = NdArray::new(&[42])?;
        let splits = a.split(0)?;
        
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].to_vec(), [42]);
        
        Ok(())
    }
    
    #[test]
    fn test_split_empty_array() -> Result<()> {
        let a = NdArray::<i32>::zeros((0, 2))?;
        let splits = a.split(0)?;
        
        assert!(splits.is_empty()); 
        Ok(())
    }

    #[test]
    fn test_repeat_1d() -> Result<()> {
        let a = NdArray::new(&[1, 2, 3])?;
        let b = a.repeat(3)?; // repeat each dimension 3 times
        assert_eq!(b.dims(), [3 * 3]); // shape: [9]
        assert_eq!(b.to_vec(), [1, 2, 3, 1, 2, 3, 1, 2, 3]);
        Ok(())
    }

    #[test]
    fn test_repeat_2d() -> Result<()> {
        let a = NdArray::new(&[[1, 2], [3, 4]])?;
        let b = a.repeat((2, 3))?; // repeat 2 times along axis 0, 3 times along axis 1
        assert_eq!(b.dims(), [4, 6]);
        assert_eq!(
            b.to_vec(),
            [
                1, 2, 1, 2, 1, 2,
                3, 4, 3, 4, 3, 4,
                1, 2, 1, 2, 1, 2,
                3, 4, 3, 4, 3, 4
            ]
        );
        Ok(())
    }

    #[test]
    fn test_repeat_dim() -> Result<()> {
        let a = NdArray::new(&[1, 2, 3])?;
        let b = a.repeat_dim(0, 2)?; // repeat along axis 0 two times
        assert_eq!(b.dims(), [6]);
        assert_eq!(b.to_vec(), [1, 2, 3, 1, 2, 3]);

        let c = a.repeat_dim(0, 1)?; // repeat once -> same as clone
        assert_eq!(c.dims(), [3]);
        assert_eq!(c.to_vec(), [1, 2, 3]);

        Ok(())
    }

    #[test]
    fn test_repeat_high_dim() -> Result<()> {
        let a = NdArray::new(&[[1, 2], [3, 4]])?;
        let b = a.repeat((2, 3))?; // more dims than array, extra dims should be treated as 1
        assert_eq!(b.dims(), [4, 6]);
        Ok(())
    }
}