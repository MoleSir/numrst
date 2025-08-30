use crate::{DTypeConvert, Error, Result, WithDType};
use super::NdArray;

impl<T: WithDType> NdArray<T> {
    pub fn copy(&self) -> Self {
        let storage = self.storage().copy(self.layout());
        Self::from_storage(storage, self.shape())
    }

    pub fn copy_from(&self, source: &Self) -> Result<()> {
        if self.shape() != source.shape() {
            Err(Error::ShapeMismatchCopyFrom { dst: self.shape().clone(), src: source.shape().clone() })?
        }

        let mut storage = self.storage_mut(0);
        for (self_storage_index, src_value) in self.layout().to_index().zip(source.iter()) {
            storage.set_unchecked(self_storage_index, src_value);
        }

        Ok(())
    }

    pub fn assign<A: AssignToNdArray<T>>(&self, source: A) -> Result<()> {
        A::assign_to(source, self)
    }
}

pub trait AssignToNdArray<T: WithDType> {
    fn assign_to(src: Self, dst: &NdArray<T>) -> Result<()>;
}

impl<T: WithDType> AssignToNdArray<T> for T {
    fn assign_to(src: T, dst: &NdArray<T>) -> Result<()> {
        let mut storage = dst.storage_mut(0);
        for storage_index in dst.layout().to_index() {
            storage.set_unchecked(storage_index, src);
        }

        Ok(())
    }
}

impl<T: WithDType> AssignToNdArray<T> for &NdArray<T> {
    fn assign_to(src: &NdArray<T>, dst: &NdArray<T>) -> Result<()> {
        if src.shape() != dst.shape() {
            Err(Error::ShapeMismatchCopyFrom { dst: dst.shape().clone(), src: src.shape().clone() })?
        }

        let mut storage = dst.storage_mut(0);

        for (self_storage_index, src_value) in dst.layout().to_index().zip(src.iter()) {
            storage.set_unchecked(self_storage_index, src_value);
        }

        Ok(())
    }
}

impl<T: WithDType> AssignToNdArray<T> for NdArray<T> {
    fn assign_to(src: NdArray<T>, dst: &NdArray<T>) -> Result<()> {
        <&NdArray<T> as AssignToNdArray<T>>::assign_to(&src, dst)
    }
}

impl<From: WithDType> NdArray<From> {
    pub fn to_dtype<To: WithDType>(&self) -> NdArray<To> 
    where
        From: DTypeConvert<To>,
    {
        let storage = self.storage().copy_map(self.layout(), From::convert);
        NdArray::<To>::from_storage(storage, self.shape())
    }
}

#[cfg(test)]
mod tests {
    use crate::IndexOp;

    use super::*;

    #[test]
    fn test_assign() {
        let a = NdArray::new(&[[1, 2, 3], [3, 4, 5], [4, 5, 6]]).unwrap();
        a.index(0).unwrap().assign(100).unwrap();
        println!("{}", a);
        a.index((1, 1)).unwrap().assign(200).unwrap();
        println!("{}", a);
        a.index((1.., 1..)).unwrap().assign(999).unwrap();
        println!("{}", a);
    }

    #[test]
    fn test_copy_1d() {
        let a = NdArray::new(&[1, 2, 3]).unwrap();
        let b = a.copy();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.to_vec(), b.to_vec());
    }

    #[test]
    fn test_copy_2d() {
        let a = NdArray::new(&[[1, 2], [3, 4]]).unwrap();
        let b = a.copy();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.to_vec(), b.to_vec());
    }

    #[test]
    fn test_to_dtype_i32_to_f32() {
        let a = NdArray::new(&[1i32, 2, 3]).unwrap();
        let b: NdArray<f32> = a.to_dtype();
        assert_eq!(b.shape(), a.shape());
        let expected = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        assert!(b.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_to_dtype_f64_to_i32() {
        let a = NdArray::new(&[1.5f64, 2.7, 3.0]).unwrap();
        let b: NdArray<i32> = a.to_dtype();
        assert_eq!(b.shape(), a.shape());
        let expected = NdArray::new(&[1i32, 2, 3]).unwrap(); // cast truncates
        assert_eq!(b.to_vec(), expected.to_vec());
    }

    #[test]
    fn test_to_dtype_2d() {
        let a = NdArray::new(&[[1i32, 2], [3, 4]]).unwrap();
        let b: NdArray<f64> = a.to_dtype();
        assert_eq!(b.shape(), a.shape());
        let expected = NdArray::new(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
        assert!(b.allclose(&expected, 1e-12, 1e-12));
    }

    #[test]
    fn test_copy_vs_to_dtype() {
        let a = NdArray::new(&[1i32, 2, 3]).unwrap();
        let b = a.copy();
        let c: NdArray<f32> = a.to_dtype();

        assert_eq!(b.to_vec(), a.to_vec());

        let expected = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }
}
