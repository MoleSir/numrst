use numrst::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arr = NdArray::<f32>::arange(1.0, 10.0)?.reshape((3, 3))?;

    let sum = arr.sum();
    let prod = arr.product();
    let min = arr.min();
    let max = arr.max();
    let mean = arr.mean();
    let var = arr.var();
    let std = arr.std();

    println!("arr:\n{}", arr);
    println!("sum = {}", sum);
    println!("product = {}", prod);
    println!("min = {}", min);
    println!("max = {}", max);
    println!("mean = {:?}", mean);
    println!("var = {:?}", var);
    println!("std = {:?}", std);

    Ok(())
}
