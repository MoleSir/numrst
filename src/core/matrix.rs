use std::sync::Arc;
use crate::{Error, NdArray, Result, Storage, WithDType};
use super::{Layout, NdArrayId, NdArrayImpl, NumDType, Shape, StorageArc, StorageRef, Vector, VectorLayout, VectorView};

pub struct MatrixLayout {
    pub(crate) shape: (usize, usize),
    pub(crate) strides: (usize, usize),
    pub(crate) start_offset: usize,
}

pub struct Matrix<T: WithDType> {
    pub(crate) storage: StorageArc<T>,
    pub(crate) layout: MatrixLayout,
}

pub struct MatrixView<'a, T: WithDType> {
    pub(crate) storage: StorageRef<'a, T>,
    pub(crate) shape: (usize, usize),
    pub(crate) strides: (usize, usize),
}

impl<T: NumDType> Matrix<T> {
    pub fn zeros(row: usize, col: usize) -> Result<Self> {
        let vec = vec![T::zero(); row * col];
        let storage = Storage::new(vec);
        Ok(Self::from_storage(storage, row, col))
    }

    pub fn zero_like(&self) -> Result<Self> {
        Self::zeros(self.row_size(), self.col_size())
    }

    pub fn eye(size: usize) -> Result<Self> {
        let mut vec = vec![T::zero(); size * size];
        for n in 0..size {
            vec[n * size + n] = T::one();
        }
        let storage = Storage::new(vec);
        Ok(Self::from_storage(storage, size, size))
    }

    pub fn add_assign(&self, row: usize, col: usize, value: T) -> Result<()> {
        let index = self.layout.get_storage_index(row, col)?;

        let mut storage = self.storage.write();
        let data = storage.data_mut(); 
        data[index] += value;
        Ok(())
    }
}

impl<T: WithDType> Matrix<T> {
    pub fn to_ndarray(&self) -> NdArray<T> {
        let shape: Shape = (self.layout.row_size(), self.layout.col_size()).into();
        let stride = [self.layout.row_stride(), self.layout.col_stride()].to_vec();
        let layout = Layout {
            shape, stride, start_offset: self.layout.start_offset
        };
        let arr = NdArrayImpl {
            id: NdArrayId::new(),
            storage: self.storage.clone(),
            layout
        };
        NdArray(Arc::new(arr))
    }

    pub fn from_ndarray(array: &NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;

        let layout = MatrixLayout {
            shape: (array.layout().dims()[0], array.layout().dims()[1]),
            strides: (array.layout().stride()[0], array.layout().stride()[1]),
            start_offset: array.layout().start_offset(),
        };

        Ok(Self {
            storage: array.storage_clone(),
            layout,
        })
    }

    pub fn to_view<'a>(&'a self) -> MatrixView<'a, T> {
        MatrixView { 
            storage: self.storage.get_ref(self.layout.start_offset), 
            shape: self.layout.shape, 
            strides: self.layout.strides, 
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Result<T> {
        let index = self.layout.get_storage_index(row, col)?;
        Ok(self.storage.get_unchecked(index))
    }

    /// Uncheck `get`
    #[inline]
    pub fn g(&self, row: usize, col: usize) -> T {
        let storage_index = self.layout.get_storage_index_uncheck(row, col);
        self.storage.get_unchecked(storage_index)
    }

    pub fn set(&self, row: usize, col: usize, value: T) -> Result<()> {
        let storage_index = self.layout.get_storage_index(row, col)?;
        self.storage.set_unchecked(storage_index, value);
        Ok(())
    }

    /// Uncheck `set`
    #[inline]
    pub fn s(&self, row: usize, col: usize, value: T) {
        let storage_index = self.layout.get_storage_index_uncheck(row, col);
        self.storage.set_unchecked(storage_index, value);
    }

    pub fn row(&self, row: usize) -> Result<Vector<T>> {
        let layout = self.layout.get_row_vector_layout(row)?;
        let storage = self.storage.clone();
        Ok(Vector { storage, layout })
    }

    pub fn col(&self, col: usize) -> Result<Vector<T>> {
        let layout = self.layout.get_col_vector_layout(col)?;
        let storage = self.storage.clone();
        Ok(Vector { storage, layout })
    }

    pub fn swap_rows(&self, r1: usize, r2: usize) -> Result<()> {
        if r1 == r2 {
            return Ok(());
        }

        let row1 = self.row(r1)?;
        let row2 = self.row(r2)?;
        row1.swap(&row2)?;

        Ok(())        

    }

    pub fn swap_rows_partial(&self, r1: usize, r2: usize, size: usize) -> Result<()> {
        if r1 == r2 {
            return Ok(());
        }

        let row1 = self.row(r1)?.take(size)?;
        let row2 = self.row(r2)?.take(size)?;
        row1.swap(&row2)?;

        Ok(())        
    }

    pub fn row_size(&self) -> usize {
        self.layout.row_size()
    }

    pub fn col_size(&self) -> usize {
        self.layout.col_size()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.row_size(), self.col_size())
    }

    pub(crate) fn from_storage(storage: Storage<T>, row: usize, col: usize) -> Self {
        Self {
            storage: StorageArc::new(storage),
            layout: MatrixLayout::contiguous(row, col)
        }
    }
}

impl<'a, T: WithDType> MatrixView<'a, T> {
    pub fn from_ndarray(array: &'a NdArray<T>) -> Result<Self> {
        let _ = array.dims2()?;

        Ok(Self {
            storage: array.storage_ref(array.layout().start_offset()),
            shape: (array.layout().dims()[0], array.layout().dims()[1]),
            strides: (array.layout().stride()[0], array.layout().stride()[1]),
        })
    }

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

    pub fn is_square(&self) -> bool {
        let (m, n) = self.shape();
        m == n 
    }

    pub fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.row_size() || col >= self.col_size() {
            None
        } else {
            Some(self.storage.get_unchecked(self.storage_index(row, col)))
        }
    }

    pub fn copy(&self) -> Matrix<T> {
        let data: Vec<_> = (0..self.row_size() * self.col_size()).into_iter()
            .map(|index| self.storage.get_unchecked(index * self.col_stride()))
            .collect();
        let storage = Storage::new(data);
        Matrix {
            storage: StorageArc::new(storage),
            layout: MatrixLayout::contiguous(self.row_size(), self.col_size())
        }
    }

    /// Uncheck `get`
    #[inline]
    pub fn g(&self, row: usize, col: usize) -> T {
        self.storage.get_unchecked(self.storage_index(row, col))
    }

    pub fn row_size(&self) -> usize {
        self.shape.0
    }

    pub fn col_size(&self) -> usize {
        self.shape.1
    }

    pub fn row_stride(&self) -> usize {
        self.strides.0
    }

    pub fn col_stride(&self) -> usize {
        self.strides.1
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.row_size(), self.col_size())
    }

    pub fn storage_index(&self, row: usize, col: usize) -> usize {
        self.strides.0 * row + self.strides.1 * col
    }
}

impl MatrixLayout {
    fn contiguous(row: usize, col: usize) -> Self {
        Self {
            shape: (row, col),
            strides: (col, 1),
            start_offset: 0,
        }
    }

    fn row_size(&self) -> usize {
        self.shape.0 
    }

    fn col_size(&self) -> usize {
        self.shape.1 
    }

    fn row_stride(&self) -> usize {
        self.strides.0 
    }

    fn col_stride(&self) -> usize {
        self.strides.1 
    }

    #[inline]
    fn get_storage_index_uncheck(&self, row: usize, col: usize) -> usize {
        self.start_offset + self.row_stride() * row + self.col_stride() * col
    }

    fn get_storage_index(&self, row: usize, col: usize) -> Result<usize> {
        if row > self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "row", len: self.row_size(), index: row });
        }
        if col > self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "col", len: self.col_size(), index: col });
        } 
        Ok(self.start_offset + self.row_stride() * row + self.col_stride() * col)
    }

    fn get_row_vector_layout(&self, row: usize) -> Result<VectorLayout> {
        if row >= self.row_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "row", len: self.row_size(), index: row });
        } else {
            Ok(VectorLayout {
                len: self.col_size(),
                stride: self.col_stride(),
                start_offset: self.start_offset + row * self.row_stride(),
            })
        }
    }

    fn get_col_vector_layout(&self, col: usize) -> Result<VectorLayout> {
        if col >= self.col_size() {
            return Err(Error::MatrixIndexOutOfRange { position: "col", len: self.col_size(), index: col });
        } else {
            Ok(VectorLayout {
                len: self.row_size(),
                stride: self.row_stride(),
                start_offset: self.start_offset + col * self.col_stride(),
            })
        }
    }
}

pub trait ToMatrixView<T: WithDType> {
    fn to_matrix_view<'a>(&'a self) -> Result<MatrixView<'a, T>>;
}

impl<T: WithDType> ToMatrixView<T> for NdArray<T> {
    fn to_matrix_view(&self) -> Result<MatrixView<'_, T>> {
        MatrixView::from_ndarray(self)
    }
}

impl<T: WithDType> ToMatrixView<T> for &NdArray<T> {
    fn to_matrix_view(&self) -> Result<MatrixView<'_, T>> {
        MatrixView::from_ndarray(self)
    }
}

impl<T: WithDType> ToMatrixView<T> for Matrix<T> {
    fn to_matrix_view(&self) -> Result<MatrixView<'_, T>> {
        Ok(self.to_view())
    }
}

impl<T: WithDType> ToMatrixView<T> for &Matrix<T> {
    fn to_matrix_view(&self) -> Result<MatrixView<'_, T>> {
        Ok(self.to_view())
    }
}

impl<T: WithDType> ToMatrixView<T> for MatrixView<'_, T> {
    fn to_matrix_view<'a>(&'a self) -> Result<MatrixView<'a, T>> {
        Ok(self.clone())
    }
}
