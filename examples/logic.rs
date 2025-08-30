use numrst::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Comparison ops ---
    let a = NdArray::<i32>::arange(0, 9)?.reshape((3, 3))?;
    let b = NdArray::<i32>::fill((3, 3), 4)?;

    let eq = a.eq(&b)?;
    let ne = a.ne(&b)?;
    let lt = a.lt(&b)?;
    let ge = a.ge(&b)?;

    println!("a:\n{}", a);
    println!("b:\n{}", b);
    println!("a == b:\n{}", eq);
    println!("a != b:\n{}", ne);
    println!("a < b:\n{}", lt);
    println!("a >= b:\n{}", ge);

    // --- Boolean logic ops ---
    let x = NdArray::<bool>::trues((2, 2))?;
    let y = NdArray::<bool>::falses((2, 2))?;

    let and = x.and(&y)?;
    let or = x.or(&y)?;
    let xor = x.xor(&y)?;

    println!("\nx:\n{}", x);
    println!("y:\n{}", y);
    println!("x AND y:\n{}", and);
    println!("x OR y:\n{}", or);
    println!("x XOR y:\n{}", xor);

    // --- Using select (if-else like) ---
    let cond = a.lt(&b)?;
    let chosen = NdArray::<i32>::select(&cond, &a, &b)?;

    println!("\nCondition (a < b):\n{}", cond);
    println!("Select result (if a<b then a else b):\n{}", chosen);

    Ok(())
}
