use crate::{Error, FloatDType, NdArray, NumDType, Result, StorageMut, StorageRef, WithDType};
use super::{AsMatrixView, AsMatrixViewFloat, AsMatrixViewMut, AsMatrixViewNum, VectorView};

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
        assert_eq!(view.diag(), [1, 4]);
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
