use numrst::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create arrays with different constructors
    let zeros = NdArray::<f32>::zeros((2, 3))?;
    let ones = NdArray::<f32>::ones((2, 3))?;
    let arange = NdArray::<i32>::arange(0, 10)?;
    let rand = NdArray::<f32>::rand(0.0, 1.0, (2, 2))?;
    let randn = NdArray::<f32>::randn(0.0, 1.0, (2, 2))?;

    println!("zeros:\n{}", zeros);
    println!("ones:\n{}", ones);
    println!("arange:\n{}", arange);
    println!("rand:\n{}", rand);
    println!("randn:\n{}", randn);

    Ok(())
}
