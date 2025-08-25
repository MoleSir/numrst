use numrs::{IndexOp, NdArray};

#[allow(unused)]
fn result_main() -> Result<(), Box<dyn std::error::Error>> {
    let arr = NdArray::new(1)?;
    println!("{}", arr.shape());
    println!("{}", arr);

    let arr = NdArray::new(&[1.0f32, 2., 3.])?;
    println!("{}", arr);
    println!("{}", arr.dtype());

    let arr = NdArray::new(&[[1, 2, 3], [3, 4, 5]])?;
    println!("{}", arr.shape());

    let arr = NdArray::<f32>::ones(1)?;
    println!("{} {}", arr.shape(), arr.dtype());

    let arr = NdArray::randn(0.0f64, 1., (1, 2, 3))?;
    println!("{} {}", arr.shape(), arr.dtype());

    let arr = NdArray::fill((2, 3, 4), 1.2)?;
    println!("{} {}", arr.shape(), arr.dtype());

    let arr = NdArray::fill((10, 4, 4), 100u32)?;
    println!("{} {}", arr.shape(), arr.dtype());

    let arr = NdArray::arange(0., 10.)?;
    println!("{}", arr.shape());
    println!("{}", arr.index(0)?.to_scalar()?);
    println!("{}", arr.index(9)?.to_scalar()?);

    let ts = NdArray::trues((3, 4))?;
    println!("{}", ts);

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{e}");
    }
}