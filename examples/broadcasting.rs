use numrst::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = NdArray::<f32>::ones((3, 3))?;
    let b = NdArray::<f32>::arange(0.0, 3.0)?.reshape((3, 1))?;

    let c = a.broadcast_add(&b)?; // broadcasting
    println!("a:\n{}", a);
    println!("b:\n{}", b);
    println!("a + b (broadcast):\n{}", c);

    Ok(())
}
