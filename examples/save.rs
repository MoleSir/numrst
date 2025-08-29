use numrst::NdArray;

#[allow(unused)]
fn result_main() -> Result<(), Box<dyn std::error::Error>> {
    let arr = NdArray::<f32>::randn(0., 1., (4, 5, 9)).unwrap();
    arr.save_file("./data/nrst/test1.nrst")?;
    let bs = NdArray::new(&[[true, false], [false, true]])?;
    bs.save_file("./data/nrst/test2.nrst")?;

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{e}");
    }
}