use numrs::{DType, IndexOp, NdArray};

#[allow(unused)]
fn result_main() -> Result<(), Box<dyn std::error::Error>> {
    let arr = NdArray::new(1)?;
    println!("{:?}", arr.shape());

    let arr = NdArray::new(&[1.0f32, 2., 3.])?;
    println!("{:?}", arr.dtype());

    let arr = NdArray::new(&[[1, 2, 3], [3, 4, 5]])?;
    println!("{:?}", arr.shape());

    let arr = NdArray::ones(1, DType::I32)?;
    println!("{:?} {:?}", arr.shape(), arr.dtype());

    let arr = NdArray::randn(0., 1., (1, 2, 3), DType::F64)?;
    println!("{:?} {:?}", arr.shape(), arr.dtype());

    let arr = NdArray::fill((2, 3, 4), 1.2)?;
    println!("{:?} {:?}", arr.shape(), arr.dtype());

    let arr = NdArray::fill((10, 4, 4), 100u32)?;
    println!("{:?} {:?}", arr.shape(), arr.dtype());

    let arr = NdArray::arange(0., 10.)?;
    println!("{:?}", arr.shape());
    println!("{:?}", arr.index(0)?.to_scalar()?);
    println!("{:?}", arr.index(9)?.to_scalar()?);
    

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{e}");
    }
}