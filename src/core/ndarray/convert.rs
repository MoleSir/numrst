use crate::{DTypeConvert, WithDType};
use super::NdArray;

impl<T: WithDType> NdArray<T> {
    pub fn copy(&self) -> Self {
        let storage = self.storage().copy(self.layout());
        Self::from_storage(storage, self.shape())
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
    use super::*;

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
