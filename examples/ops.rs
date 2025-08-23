use numrs::NdArray;

#[allow(unused)]
fn result_main() -> Result<(), Box<dyn std::error::Error>> {
    let a = NdArray::new(&[1.0f32, 2., 3.])?;
    let b = NdArray::new(&[1.0f32, 2., 3.])?;
    let c = a.add(&b)?;

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{e}");
    }
}