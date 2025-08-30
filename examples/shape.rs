use numrst::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Reshape ---
    let a = NdArray::<i32>::arange(0, 12)?;
    let a2d = a.reshape((3, 4))?;
    let a3d = a.reshape((2, 2, 3))?;

    println!("Original a (1D):\n{}", a);
    println!("Reshaped a (3x4):\n{}", a2d);
    println!("Reshaped a (2x2x3):\n{}", a3d);

    // --- Transpose ---
    let t = a2d.transpose(0, 1)?;
    println!("Transpose a2d (swap dim 0 <-> 1):\n{}", t);

    // --- Unsqueeze & Squeeze ---
    let b = NdArray::<f32>::arange(0.0, 6.0)?.reshape((2, 3))?;
    let b_unsq = b.unsqueeze(0)?; // add axis at front -> (1,2,3)
    let b_sq = b_unsq.squeeze(0)?; // remove axis -> (2,3)

    println!("b shape: {}", b.shape());
    println!("b_unsq shape: {}", b_unsq.shape());
    println!("b_sq shape: {}", b_sq.shape());

    // --- Narrow & Narrow Range ---
    let narrowed = b.narrow(1, 0, 2)?; // take first 2 cols along axis 1
    let narrowed_range = b.narrow_range(1, &rng!(0:2))?;

    println!("narrowed (axis=1, first 2 cols):\n{}", narrowed);
    println!("narrowed_range (axis=1, 0..2):\n{}", narrowed_range);

    // --- Concatenate ---
    let c1 = NdArray::<i32>::full((2, 2), 1)?;
    let c2 = NdArray::<i32>::full((2, 2), 2)?;
    let cat = NdArray::cat(&[&c1, &c2], 0)?; // concat along axis=0

    println!("Concatenate along axis=0:\n{}", cat);

    // --- Stack ---
    let s1 = NdArray::<i32>::full((2, 2), 3)?;
    let s2 = NdArray::<i32>::full((2, 2), 4)?;
    let stacked = NdArray::stack(&[&s1, &s2], 0)?; // stack new axis

    println!("Stacked along new axis=0:\n{}", stacked);

    Ok(())
}
