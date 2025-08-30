use numrst::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = NdArray::<f32>::ones((2, 2))?;
    let b = NdArray::<f32>::full((2, 2), 2.0)?;

    let c = a.add(&b)?;
    let d = b.sub(&a)?;
    let e = b.mul(&c)?;
    let f = e.div(&d)?;

    println!("a:\n{:?}", a);
    println!("b:\n{}", b);
    println!("a + b:\n{}", c);
    println!("b - a:\n{:?}", d);
    println!("b * (a + b):\n{}", e);
    println!("e / d:\n{}", f);

    Ok(())
}
