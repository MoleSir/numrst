use super::{DType, WithDType};

impl WithDType for bool {
    const DTYPE: DType = DType::Bool;

    fn dtype() -> DType {
        DType::F32
    }
}