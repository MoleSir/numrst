use numrst::{view::AsMatrixViewMut, IndexOp, NdArray};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total = NdArray::<f32>::zeros((5, 5)).unwrap();

    {
        let mut sub = total.index((1..3, 2..4)).unwrap();
        let source = sub.randn_like(0.0, 1.0).unwrap();
        let mut sub_view = sub.matrix_view_mut().unwrap();

        sub_view.copy_from(&source).unwrap();
    }
    
    println!("{}", total);

    Ok(())
}
