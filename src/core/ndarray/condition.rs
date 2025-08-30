use crate::{Error, Result, Shape, Storage, WithDType};
use super::NdArray;

impl<T: WithDType> NdArray<T> {
    pub fn filter(&self, conditions: &NdArray<bool>) -> Result<NdArray<T>> {
        // self should has the same shape the condition
        if self.dims() != conditions.dims() {
            return Err(Error::ShapeMismatchFilter { src: self.shape().clone(), condition: conditions.shape().clone() });
        }
        
        let vec: Vec<_> = self.iter().zip(conditions.iter())
            .filter(|(_, condition)| *condition)
            .map(|(value, _)| value)
            .collect();
        let shape = vec.len();

        let storage = Storage::new(vec);
        Ok(Self::from_storage(storage, shape))
    }

    pub fn select<TV, FV>(mask: &NdArray<bool>, true_val: TV, false_val: FV) -> Result<Self> 
    where 
        TV: ConditionValue<T>,
        FV: ConditionValue<T>,
    {
        if !true_val.check_shape(mask.shape()) {
            Err(Error::ShapeMismatchSelect { mask: mask.shape().clone(), who: "true_val", })?
        }

        if !false_val.check_shape(mask.shape()) {
            Err(Error::ShapeMismatchSelect { mask: mask.shape().clone(), who: "false_val", })?
        }

        let result = true_val.copy_self(mask.shape())?;
        assert_eq!(mask.dims(), result.dims());

        {
            let mut result_storage = result.storage_mut(0);
            // zip ( result storage index, condition and false value )
            for ((result_index, condition), fv) in result.layout().to_index().zip(mask.iter()).zip(false_val.iter_value()) {
                if !condition {
                    result_storage.set_unchecked(result_index, fv);
                }
            }
        }
        
        Ok(result)
    }
}

pub trait ConditionValue<T: WithDType> {
    fn check_shape(&self, shape: &Shape) -> bool;
    fn copy_self(&self, shape: &Shape) -> Result<NdArray<T>>;
    fn iter_value(&self) -> impl Iterator<Item = T>;
}

impl<T: WithDType> ConditionValue<T> for &NdArray<T> {
    fn check_shape(&self, shape: &Shape) -> bool {
        self.shape() == shape
    }

    fn copy_self(&self, shape: &Shape) -> Result<NdArray<T>> {
        assert_eq!(self.shape(), shape);
        Ok(self.copy())
    }

    fn iter_value(&self) -> impl Iterator<Item = T> {
        self.iter()
    }
}

impl<T: WithDType> ConditionValue<T> for T {
    fn check_shape(&self, _: &Shape) -> bool {
        true
    }

    fn copy_self(&self, shape: &Shape) -> Result<NdArray<T>> {
        NdArray::<T>::fill(shape, *self)
    }

    fn iter_value(&self) -> impl Iterator<Item = T> {
        std::iter::repeat(*self)
    }
}

#[cfg(test)]
mod test {
    use crate::NdArray;

    #[test]
    fn test_filter_basic() {
        let a = NdArray::new(&[10, 20, 30, 40, 50]).unwrap();
        let mask = NdArray::new(&[false, true, false, true, true]).unwrap();

        let result = a.filter(&mask).unwrap();
        assert_eq!(result.to_vec(), [20, 40, 50]);
    }

    #[test]
    fn test_filter_all_false() {
        let a = NdArray::new(&[1, 2, 3]).unwrap();
        let mask = NdArray::new(&[false, false, false]).unwrap();

        let result = a.filter(&mask).unwrap();
        assert!(result.to_vec().is_empty());
    }

    #[test]
    fn test_filter_shape_mismatch() {
        let a = NdArray::new(&[1, 2, 3, 4]).unwrap();
        let mask = NdArray::new(&[true, false]).unwrap();

        let result = a.filter(&mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_2d_flatten_order() {
        let a = NdArray::new(&[[1, 2, 3],
                               [4, 5, 6]]).unwrap();
        let mask = NdArray::new(&[[true,  false, true],
                                  [false, true,  false]]).unwrap();

        let result = a.filter(&mask).unwrap();
        assert_eq!(result.to_vec(), [1, 3, 5]);
    }

    #[test]
    fn test_select_scalar_values() {
        let mask = NdArray::new(&[true, false, true, false]).unwrap();
        let result = NdArray::<i32>::select(&mask, 1, 0).unwrap();
        assert_eq!(result.to_vec(), [1, 0, 1, 0]);
    }
    
    #[test]
    fn test_select_array_values() {
        let mask = NdArray::new(&[true, false, true, false]).unwrap();
        
        let true_vals = NdArray::new(&[10, 20, 30, 40]).unwrap();
        let false_vals = NdArray::new(&[100, 200, 300, 400]).unwrap();
        
        let result = NdArray::<i32>::select(&mask, &true_vals, &false_vals).unwrap();
        assert_eq!(result.to_vec(), [10, 200, 30, 400]);
    }
    
    #[test]
    fn test_select_mixed_values() {
        let mask = NdArray::new(&[true, false, true, false]).unwrap();
        
        let true_vals = 5;  // 标量
        let false_vals = NdArray::new(&[100, 200, 300, 400]).unwrap();
        
        let result = NdArray::<i32>::select(&mask, true_vals, &false_vals).unwrap();
        assert_eq!(result.to_vec(), [5, 200, 5, 400]);
    }
    
    #[test]
    fn test_select_shape_mismatch() {
        let mask = NdArray::new(&[true, false, true]).unwrap();
        let true_vals = NdArray::new(&[1, 2, 3, 4]).unwrap();
        let false_vals = 0;
        
        let result = NdArray::<i32>::select(&mask, &true_vals, false_vals);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_select_all_true_or_all_false() {
        let mask = NdArray::new(&[true, true, true]).unwrap();
        let result = NdArray::<i32>::select(&mask, 1, 0).unwrap();
        assert_eq!(result.to_vec(), [1, 1, 1]);
    
        let mask = NdArray::new(&[false, false, false]).unwrap();
        let result = NdArray::<i32>::select(&mask, 1, 0).unwrap();
        assert_eq!(result.to_vec(), [0, 0, 0]);
    }
    
    #[test]
    fn test_filter_2d_all_true() {
        let a = NdArray::new(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]).unwrap();
        let mask = NdArray::new(&[[true, true, true], [true, true, true], [true, true, true]]).unwrap();
    
        let result = a.filter(&mask).unwrap();
        assert_eq!(result.to_vec(), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
    
    #[test]
    fn test_filter_3d_partial_true() {
        let a = NdArray::new(&[
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]).unwrap();
        let mask = NdArray::new(&[
            [[true, false], [false, true]],
            [[false, true], [true, false]]
        ]).unwrap();
    
        let result = a.filter(&mask).unwrap();
        assert_eq!(result.to_vec(), [1, 4, 6, 7]);
    }
    
    #[test]
    fn test_select_2d_array_values() {
        let mask = NdArray::new(&[[true, false, true], [false, true, false]]).unwrap();
        let true_vals = NdArray::new(&[[10, 20, 30], [40, 50, 60]]).unwrap();
        let false_vals = NdArray::new(&[[100, 200, 300], [400, 500, 600]]).unwrap();
    
        let result = NdArray::<i32>::select(&mask, &true_vals, &false_vals).unwrap();
        assert_eq!(result.to_vec(), [10, 200, 30, 400, 50, 600]);
    }
    
    #[test]
    fn test_select_3d_mixed_values() {
        let mask = NdArray::new(&[
            [[true, false], [false, true]],
            [[true, true], [false, false]]
        ]).unwrap();
        let true_val = 1;  // 标量
        let false_vals = NdArray::new(&[
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]]
        ]).unwrap();
    
        let result = NdArray::<i32>::select(&mask, true_val, &false_vals).unwrap();
        assert_eq!(result.to_vec(), [1, 20, 30, 1, 1, 1, 70, 80]);
    }
    
    #[test]
    fn test_select() {
        let scores = NdArray::new(&[
            [45., 12., 34., 90.],
            [31., 19., 84., 60.],
            [55., 34., 44., 82.],
            [85., 89., 54., 67.],
        ]).unwrap();

        // scores > 60 & scores < 85
        let mask = scores.ge(60.).unwrap().and(&scores.le(85.).unwrap()).unwrap();

        let selected_scores = NdArray::select(&mask, &scores, -1.).unwrap();
        println!("{}", selected_scores);
    }
}