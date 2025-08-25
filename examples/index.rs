use numrst::{rng, IndexOp, NdArray, Range};

#[allow(unused)]
fn result_main() -> Result<(), Box<dyn std::error::Error>> {
    let arr = NdArray::<u32>::zeros((5, 5, 5))?;

    let sub_arr = arr.index(1)?;
    println!("{:?}", sub_arr.shape());

    let sub_arr = arr.index(2)?;
    println!("{:?}", sub_arr.shape());

    let sub_arr = arr.index(rng!(1:3))?;
    println!("{:?}", sub_arr.shape());

    let sub_arr = arr.index((rng!(1:3), rng!(3:4), 1))?;
    println!("{:?}", sub_arr.shape());

    let sub_arr = arr.index((rng!(1:3), .., 1..2))?;
    println!("{:?}", sub_arr.shape());

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{e}");
    }
}