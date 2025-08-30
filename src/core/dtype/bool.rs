use super::{DType, DTypeConvert, WithDType};

impl WithDType for bool {
    const DTYPE: DType = DType::Bool;
}

impl DTypeConvert<u8> for bool { fn convert(self) -> u8 { if self { 1 } else { 0 } } }
impl DTypeConvert<i32> for bool { fn convert(self) -> i32 { if self { 1 } else { 0 } } }
impl DTypeConvert<u32> for bool { fn convert(self) -> u32 { if self { 1 } else { 0 } } }
impl DTypeConvert<usize> for bool { fn convert(self) -> usize { if self { 0 } else { 0 }} }
impl DTypeConvert<f32> for bool { fn convert(self) -> f32 { if self { 1.0 } else { 0.0 } } }
impl DTypeConvert<f64> for bool { fn convert(self) -> f64 { if self { 1.0 } else { 0.0 } } }
impl DTypeConvert<bool> for bool { fn convert(self) -> bool { self } }
