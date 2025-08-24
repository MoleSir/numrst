use std::fmt;

use crate::Storage;

use super::NdArray;

impl fmt::Display for NdArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let impl_ref = &self.0;
        let shape = self.dims();

        if shape.is_empty() {
            return write!(f, "[]");
        }

        match &*impl_ref.storage.read().unwrap() {
            Storage::U32(data) => fmt_ndarray(f, data, shape),
            Storage::I32(data) => fmt_ndarray(f, data, shape),
            Storage::F32(data) => fmt_ndarray(f, data, shape),
            Storage::F64(data) => fmt_ndarray(f, data, shape),
        }
    }
}

fn fmt_ndarray<T: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    data: &[T],
    shape: &[usize],
) -> fmt::Result {
    fn fmt_recursive<T: fmt::Display>(
        f: &mut fmt::Formatter<'_>,
        data: &[T],
        shape: &[usize],
        dim: usize,
        offset: usize,
    ) -> fmt::Result {
        if dim == shape.len() - 1 {
            write!(f, "[")?;
            for i in 0..shape[dim] {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", data[offset + i])?;
            }
            write!(f, "]")
        } else {
            write!(f, "[")?;
            let stride = shape[dim + 1..].iter().product::<usize>();
            for i in 0..shape[dim] {
                if i > 0 {
                    write!(f, "\n ")?; 
                }
                fmt_recursive(f, data, shape, dim + 1, offset + i * stride)?;
            }
            write!(f, "]")
        }
    }

    fmt_recursive(f, data, shape, 0, 0)
}
