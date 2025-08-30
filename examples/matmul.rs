use numrst::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m1 = NdArray::<f32>::rand(0.0, 1.0, (2, 3))?;
    let m2 = NdArray::<f32>::rand(0.0, 1.0, (3, 4))?;
    let m3 = m1.matmul(&m2)?;

    println!("m1:\n{}", m1);
    println!("m2:\n{}", m2);
    println!("m1 @ m2:\n{}", m3);

    Ok(())
}
