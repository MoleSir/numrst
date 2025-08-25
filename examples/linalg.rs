use numrs::NdArray;

#[allow(unused)]
fn result_main() -> Result<(), Box<dyn std::error::Error>> {
    let arr = NdArray::<i32>::ones((2, 3, 12))?;
    println!("{}", numrs::linalg::norm(&arr));

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{e}");
    }
}