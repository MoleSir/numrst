use std::marker::PhantomData;
use crate::{Error, FloatDType, Indexer, NdArray, NumDType, Result, Storage, StorageMut, StorageRef, WithDType};
use super::VectorView;

pub trait AsMatrixView<'a, T: WithDType> : Sized {
    fn shape(&self) -> (usize, usize);
    fn strides(&self) -> (usize, usize);
    fn storage_get_uncheck(&self, storage_index: usize) -> T;
    fn from_ndarray(arr: &'a NdArray<T>) -> Result<Self>;

    #[inline]
    fn row_size(&self) -> usize {
        self.shape().0
    }

    #[inline]
    fn col_size(&self) -> usize {
        self.shape().1
    }

    #[inline]
    fn row_stride(&self) -> usize {
        self.strides().0
    }

    #[inline]
    fn col_stride(&self) -> usize {
        self.strides().1
    }

    fn is_square(&self) -> bool {
        let (m, n) = self.shape();
        m == n 
    }

    #[inline]
    fn storage_index(&self, row: usize, col: usize) -> usize {
        self.row_stride() * row + self.col_stride() * col
    }

    fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            Some(self.storage_get_uncheck(self.storage_index(row, col)))
        }
    }

    #[inline]
    fn g(&self, row: usize, col: usize) -> T {
        self.storage_get_uncheck(self.storage_index(row, col))
    }
    
    fn eqal(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            false 
        } else {
            for i in 0..self.row_size() {
                for j in 0..self.col_size() {
                    if self.g(i, j) != other.g(i, j) {
                        return false;
                    }
                }
            }
            true
        }
    }

    fn copy(&'a self) -> NdArray<T> {
        let data: Vec<_> = self.iter().collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, (self.row_size(), self.col_size()))
    }

    fn diag(&'a self) -> MatrixViewDiagIter<'a, T, Self> {
        MatrixViewDiagIter { 
            view: &self, 
            len: self.row_size().min(self.col_size()), 
            index: 0, 
            _marker: PhantomData,
        }
    }

    fn iter(&'a self) -> MatrixViewIter<'a, T, Self> {
        MatrixViewIter {
            _marker: PhantomData,
            view: self, 
            row: 0,
            col: 0,
        }
    }

    fn to_vec(&'a self) -> Vec<T> {
        self.iter().collect()
    }

    fn to_mat(&'a self) -> Vec<Vec<T>> {
        (0..self.row_size())
            .map(|row| {
                (0..self.col_size())
                    .map(|col| self.g(row, col))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

pub trait AsMatrixViewNum<'a, T: NumDType> : AsMatrixView<'a, T> {
    fn matmul(&self, rhs: &Self) -> Result<NdArray<T>> {
        let m = self.row_size();
        let k = self.col_size();
        let n = rhs.col_size();

        if k != rhs.row_size() {
            return Err(Error::ShapeMismatchBinaryOp { lhs: self.shape().into(), rhs: rhs.shape().into(), op: "matmul" })
        }

        let mut data = vec![T::zero(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for kk in 0..k {
                    let a = self.g(i, kk); 
                    let b = rhs.g(kk, j);
                    sum = sum + a * b;
                }
                data[i * self.col_size() + j] = sum;
            }
        }

        let storage = Storage::new(data);
        Ok(NdArray::from_storage(storage, (m, n)))
    }
}

pub trait AsMatrixViewFloat<'a, T: FloatDType> : AsMatrixView<'a, T> {
    fn norm(&'a self) -> T {
        self.iter()
            .map(|v| v.powi(2))
            .sum::<T>()
            .sqrt()
    }
}

pub trait AsMatrixViewMut<'a, T: WithDType> : AsMatrixView<'a, T> {
    fn storage_set_uncheck(&mut self, storage_index: usize, value: T);
    fn from_ndarray_mut(arr: &'a mut NdArray<T>) -> Result<Self>;

    fn set(&mut self, row: usize, col: usize, value: T) -> Option<()> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            self.storage_set_uncheck(self.storage_index(row, col), value);
            Some(())
        }
    }

    #[inline]
    fn s(&mut self, row: usize, col: usize, value: T) {
        self.storage_set_uncheck(self.storage_index(row, col), value)
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) -> Result<()> {
        if r1 == r2 {
            return Ok(());
        }
        if r1 > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_rows", len: self.row_size(), index: r1 });
        } 
        if r2 > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_rows", len: self.row_size(), index: r2 });
        } 

        let stride = self.col_stride();
        let mut start_index1 = self.col_size() * r1;
        let mut start_index2 = self.col_size() * r2;

        for _ in 0..self.col_size() {
            let value1 = self.storage_get_uncheck(start_index1);
            let value2 = self.storage_get_uncheck(start_index2);
            self.storage_set_uncheck(start_index1, value2);
            self.storage_set_uncheck(start_index2, value1);
            start_index1 += stride;
            start_index2 += stride;
        }

        Ok(())    
    }

    fn swap_cols(&mut self, c1: usize, c2: usize) -> Result<()> {
        if c1 == c2 {
            return Ok(());
        }
        if c1 > self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_cols", len: self.row_size(), index: c1 });
        } 
        if c2 > self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "swap_cols", len: self.row_size(), index: c2 });
        } 

        let stride = self.row_stride();
        let mut start_index1 = c1;
        let mut start_index2 = c2;

        for _ in 0..self.row_size() {
            let value1 = self.storage_get_uncheck(start_index1);
            let value2 = self.storage_get_uncheck(start_index2);
            self.storage_set_uncheck(start_index1, value2);
            self.storage_set_uncheck(start_index2, value1);
            start_index1 += stride;
            start_index2 += stride;
        }

        Ok(())    
    }
    
    fn copy_from(&mut self, arr: &NdArray<T>) -> Result<()> {
        let view = arr.matrix_view()?;
        self.copy_from_view(&view)
    }
    
    fn copy_from_view<'b>(&mut self, source: &impl AsMatrixView<'b, T>) -> Result<()> {
        if self.shape() != source.shape() {
            return Err(Error::ShapeMismatchMatrix {
                lhs: self.shape(),
                rhs: source.shape(),
                op: "copy_from_view",
            });
        }
    
        for i in 0..self.row_size() {
            for j in 0..self.col_size() {
                self.s(i, j, source.g(i, j));
            }
        }
    
        Ok(())
    }
}

pub struct MatrixViewIter<'a, T: WithDType, V: AsMatrixView<'a, T>> {
    view: &'a V,
    row: usize,
    col: usize,
    _marker: PhantomData<T>,
}

impl<'a, T: WithDType, V: AsMatrixView<'a, T>> Iterator for MatrixViewIter<'a, T, V> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.col == self.view.col_size() {
            self.col = 0;
            self.row += 1;
            if self.row == self.view.row_size() {
                return None;
            } 
        }

        let value = self.view.g(self.row, self.col);
        self.col += 1;
        Some(value)
    }
}

impl<'a, T: WithDType, V: AsMatrixView<'a, T>> ExactSizeIterator for MatrixViewIter<'a, T, V> {
    fn len(&self) -> usize {
        self.row * self.col
    }
}

pub struct MatrixViewDiagIter<'a, T: WithDType, V: AsMatrixView<'a, T>> {
    view: &'a V,
    len: usize,
    index: usize,
    _marker: PhantomData<T>,
}

impl<'a, T: WithDType, V: AsMatrixView<'a, T>> Iterator for MatrixViewDiagIter<'a, T, V> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.len {
            None
        } else {
            let v = self.view.g(self.index, self.index);
            self.index += 1;
            Some(v)
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
///         Matrix View 
//////////////////////////////////////////////////////////////////////////////

pub struct MatrixView<'a, T: WithDType> {
    pub(crate) storage: StorageRef<'a, T>,
    pub(crate) shape: (usize, usize),
    pub(crate) strides: (usize, usize),
}

impl<'a, T: WithDType> AsMatrixView<'a, T> for MatrixView<'a, T> {
    #[inline]
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[inline]
    fn strides(&self) -> (usize, usize) {
        self.strides
    }

    #[inline]
    fn storage_get_uncheck(&self, storage_index: usize) -> T {
        self.storage.get_unchecked(storage_index)
    }

    #[inline]
    fn from_ndarray(array: &'a NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;

        Ok(Self {
            storage: array.storage_ref(array.layout().start_offset()),
            shape: (array.layout().dims()[0], array.layout().dims()[1]),
            strides: (array.layout().stride()[0], array.layout().stride()[1]),
        })
    }
}

impl<'a, T: NumDType> AsMatrixViewNum<'a, T> for MatrixView<'a, T> {}
impl<'a, T: FloatDType> AsMatrixViewFloat<'a, T> for MatrixView<'a, T> {}

impl<'a, T: WithDType> MatrixView<'a, T> {
    pub fn clone(&'a self) -> Self {
        Self {
            shape: self.shape,
            storage: self.storage.clone(),
            strides: self.strides
        }
    }

    pub fn row(&'a self, row: usize) -> Result<VectorView<'a, T>> {
        if row > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "row", len: self.row_size(), index: row });
        } else {
            let storage = self.storage.slice(row * self.row_stride());
            Ok(VectorView {
                storage,
                len: self.col_size(),
                stride: self.col_stride(),
            })
        }
    }

    pub fn col(&'a self, col: usize) -> Result<VectorView<'a, T>> {
        if col > self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "col", len: self.col_size(), index: col });
        } else {
            let storage = self.storage.slice(col * self.col_stride());
            Ok(VectorView {
                storage,
                len: self.row_size(),
                stride: self.row_stride(),
            })
        }
    }

    pub fn slice<RI, CI>(&'a self, ri: RI, ci: CI) -> Result<Self>
    where
        RI: Into<Indexer>,
        CI: Into<Indexer>,
    {
        let ri: Indexer = ri.into();
        let ci: Indexer = ci.into();
        
        let (rbegin, rstep, rsize) = ri.begin_step_size(self.row_size(), "slice_row")?;
        let (cbegin, cstep, csize) = ci.begin_step_size(self.col_size(), "slice_col")?;
        
        let new_storage = self.storage.slice(rbegin * self.row_stride() + cbegin * self.col_stride());

        Ok(Self {
            storage: new_storage,
            shape: (rsize, csize),
            strides: (self.row_stride() * rstep, self.col_stride() * cstep),
        })
    }

    pub fn transpose(&'a self) -> Self {
        Self {
            shape: (self.col_size(), self.row_size()),
            strides: (self.col_stride(), self.row_stride()),
            storage: self.storage.clone()
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
///         Matrix View Mut
//////////////////////////////////////////////////////////////////////////////

pub struct MatrixViewMut<'a, T: WithDType> {
    pub(crate) storage: StorageMut<'a, T>,
    pub(crate) shape: (usize, usize),
    pub(crate) strides: (usize, usize),
}

impl<'a, T: WithDType> AsMatrixView<'a, T> for MatrixViewMut<'a, T> {
    #[inline]
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[inline]
    fn strides(&self) -> (usize, usize) {
        self.strides
    }

    #[inline]
    fn storage_get_uncheck(&self, storage_index: usize) -> T {
        self.storage.get_unchecked(storage_index)
    }

    #[inline]
    fn from_ndarray(array: &'a NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;

        Ok(Self {
            storage: array.storage_mut(array.layout().start_offset()),
            shape: (array.layout().dims()[0], array.layout().dims()[1]),
            strides: (array.layout().stride()[0], array.layout().stride()[1]),
        })
    }
}

impl<'a, T: FloatDType> AsMatrixViewFloat<'a, T> for MatrixViewMut<'a, T> {}

impl<'a, T: WithDType> AsMatrixViewMut<'a, T> for MatrixViewMut<'a, T> {
    fn storage_set_uncheck(&mut self, storage_index: usize, value: T) {
        self.storage.set_unchecked(storage_index, value);
    }

    fn from_ndarray_mut(array: &'a mut NdArray<T>) -> Result<Self> {
        Self::from_ndarray(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{view::{AsMatrixView, AsMatrixViewFloat, AsMatrixViewMut, AsMatrixViewNum, AsVectorView}, IndexOp, NdArray};

    #[test]
    fn test_matrix_view_basic_access() {
        let arr = NdArray::from_vec([1, 2, 3, 4], (2, 2)).unwrap();
        let view = MatrixView::from_ndarray(&arr).unwrap();

        assert_eq!(view.shape(), (2, 2));
        assert_eq!(view.g(0, 0), 1);
        assert_eq!(view.g(0, 1), 2);
        assert_eq!(view.g(1, 0), 3);
        assert_eq!(view.g(1, 1), 4);

        assert_eq!(view.get(2, 0), None); // 越界
        assert_eq!(view.diag().collect::<Vec<_>>(), [1, 4]);
    }

    #[test]
    fn test_matrix_view_row_col() {
        let arr = NdArray::from_vec([1, 2, 3, 4, 5, 6], (2, 3)).unwrap();
        let view = MatrixView::from_ndarray(&arr).unwrap();

        let row0 = view.row(0).unwrap();
        assert_eq!(row0.to_vec(), [1, 2, 3]);

        let col1 = view.col(1).unwrap();
        assert_eq!(col1.to_vec(), [2, 5]);
    }

    #[test]
    fn test_matrix_view_transpose() {
        let arr = NdArray::from_vec([1, 2, 3, 4, 5, 6], (2, 3)).unwrap();
        let view = MatrixView::from_ndarray(&arr).unwrap();
        let t = view.transpose();

        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.g(0, 0), 1);
        assert_eq!(t.g(1, 0), 2);
        assert_eq!(t.g(2, 1), 6);
    }

    #[test]
    fn test_matrix_eqal_and_copy() {
        let arr1 = NdArray::from_vec([1, 2, 3, 4], (2, 2)).unwrap();
        let arr2 = NdArray::from_vec([1, 2, 3, 4], (2, 2)).unwrap();
        let arr3 = NdArray::from_vec([5, 6, 7, 8], (2, 2)).unwrap();

        let v1 = MatrixView::from_ndarray(&arr1).unwrap();
        let v2 = MatrixView::from_ndarray(&arr2).unwrap();
        let v3 = MatrixView::from_ndarray(&arr3).unwrap();

        assert!(v1.eqal(&v2));
        assert!(!v1.eqal(&v3));

        let copy = v1.copy();
        assert_eq!(copy.to_vec(), [1, 2, 3, 4]);
    }

    #[test]
    fn test_matrix_matmul() {
        let a = NdArray::from_vec([1, 2, 3, 4], (2, 2)).unwrap();
        let b = NdArray::from_vec([5, 6, 7, 8], (2, 2)).unwrap();

        let va = MatrixView::from_ndarray(&a).unwrap();
        let vb = MatrixView::from_ndarray(&b).unwrap();

        let c = va.matmul(&vb).unwrap();
        assert_eq!(c.to_vec(), [19, 22, 43, 50]); // 正确结果
    }

    #[test]
    fn test_matrix_norm() {
        let arr = NdArray::from_vec([3.0f32, 4.0], (1, 2)).unwrap();
        let view = MatrixView::from_ndarray(&arr).unwrap();

        let norm = view.norm();
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_view_mut_set_and_swap_rows() {
        let mut arr = NdArray::from_vec([1, 2, 3, 4], (2, 2)).unwrap();
        let mut view = MatrixViewMut::from_ndarray_mut(&mut arr).unwrap();

        view.s(0, 0, 9);
        assert_eq!(view.to_vec(), [9, 2, 3, 4]);

        view.swap_rows(0, 1).unwrap();
        assert_eq!(view.to_vec(), [3, 4, 9, 2]);

        drop(view);
        assert_eq!(arr.to_vec(), [3, 4, 9, 2]);
    }

    #[test]
    fn test_matrix_view_mut_swap_cols() {
        let mut arr = NdArray::from_vec([1, 2, 3, 4, 5, 6], (2, 3)).unwrap();
        let mut view = MatrixViewMut::from_ndarray_mut(&mut arr).unwrap();

        view.swap_cols(0, 2).unwrap();
        assert_eq!(view.to_vec(), [3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn test_sub_view() {
        let total = NdArray::<f32>::zeros((5, 5)).unwrap();

        {
            let mut sub = total.index((1..3, 2..4)).unwrap();
            let source = sub.randn_like(0.0, 1.0).unwrap();
            let mut sub_view = sub.matrix_view_mut().unwrap();
    
            sub_view.copy_from(&source).unwrap();
        }
        
        println!("{}", total);
    }
}
