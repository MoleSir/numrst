use numrst::{rng, IndexOp, NdArray, Range};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arr = NdArray::<u32>::arange(0, 27)?.reshape((3, 3, 3))?;

    let single = arr.index(1)?; // single element along first axis
    let slice = arr.index(rng!(1:3))?; // range
    let mixed = arr.index((rng!(0:2), .., 1..2))?; // mixed slicing

    println!("arr:\n{}", arr);
    println!("index(1):\n{}", single);
    println!("index(rng!(1:3)):\n{}", slice);
    println!("index((rng!(0:2), .., 1..2)):\n{}", mixed);

    Ok(())
}
