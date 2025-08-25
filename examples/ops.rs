use numrst::NdArray;

#[allow(unused)]
fn result_main() -> Result<(), Box<dyn std::error::Error>> {
    let a = NdArray::new(&[1.0f32, 2., 3.])?;
    let b = NdArray::new(&[1.0f32, 2., 3.])?;
    let c = a.add(&b)?;
    let c = a.sub(&b)?;

    let a = NdArray::new(&[[1, 2, 3], [4, 5, 6]])?;
    let sum = a.sum_axis(1)?;

    let a = NdArray::<f32>::randn(0., 1., (4, 4, 5))?;
    let b = NdArray::<f32>::randn(0., 1., (4, 5, 3))?;
    let c = a.matmul(&b)?;
    println!("{}", c.shape());
    println!("{}", c);

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{e}");
    }
}