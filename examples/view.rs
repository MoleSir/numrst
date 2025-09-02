use numrst::{IndexOp, NdArray};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = NdArray::<f32>::rand(0.0, 1.0, (5, 9, 7, 11))?;
    println!("{}", a.shape());
    
    let mat_a = a.index((1, 1, 1..5, 2..9))?;
    println!("{}", mat_a.shape());
    println!("{}", mat_a);


    Ok(())
}
