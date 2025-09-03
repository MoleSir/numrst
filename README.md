# NumRst

NumRst is a Rust implementation of multi-dimensional arrays (NdArray) and numerical computation, inspired by NumPy.  

It provides a rich set of numerical operations, including broadcasting, indexing, matrix operations, reductions, logical and comparison ops, aiming to build a high-performance numerical computing tool in the Rust ecosystem.



## 🚀Features

- Data types supported:  `Bool, U8, I8, U16, I16, U32, I32, USize, F32, F64`
  
- Operators & Arithmetic:
  - Basic ops: `+ -  / minimum maximum`
  - Comparison: `eq ne lt le gt ge`
  - Logic: `and or xor`
  - Float functions: `exp sin cos sqrt tanh floor ceil round abs neg ln`

- NdArray operations:
  - Shape manipulation: `reshape`, `transpose`, `squeeze`, `unsqueeze`, `narrow`, `narrow_range`, `repeat`, `flatten`, `split`
  - Concatenation/stacking: `cat`, `stack`

- Broadcasting:  NumPy-like broadcasting (`broadcast_add`, `broadcast_sub`, etc.)
  
- Reductions:
  - Global: `sum`, `product`, `min`, `max`, `mean`, `var`, `std`
  - Along axis: `sum_axis`, `max_axis`, `argmax_axis`, etc.

- Array creation:
  - `zeros`, `ones`, `arange`, `rand`, `randn`, `trues`, `falses`
  - `from_vec`, `full`, `new`

- Indexing & slicing:
  - Single index: `arr.index(2)?`
  - Range: `arr.index(rng!(1:3))?`
  - Mixed multi-dim: `arr.index((rng!(1:3), .., 1..2))?`

- Matrix operations:
  - `matmul`

- File format: 
  - Support operate with .npy and .npz
  - Custom file format .nrst and .nrsz

- View:
  - Matrix view: `MatrixView` and `MatrixViewMut`
  - Matrix view: `MatrixView` and `MatrixViewMut`
  - Unsafe view: `MatrixViewUsf` and `VectorViewUsf` are unsafe view, use ptr as storage

- Linear algebra:
  - basic: `dot`, `matmul`, `trace`
  - decompose: lu, qr, cholesky, bidiagonal
  - solve: lu, qr, cholesky, eig(only support symmetric), svd(can't rank_deficient)

## Quick Start

```rust
use numrst::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 5x5x5 zero array
    let arr = NdArray::<u32>::zeros((5, 5, 5))?;

    // Basic arithmetic
    let b = NdArray::<u32>::ones((5, 5, 5))?;
    let c = arr.add(&b)?;
    
    // Indexing and slicing
    let sub_arr = c.index((rng!(1:3), .., 2))?;
    
    // Matrix multiplication
    let m1 = NdArray::<f32>::rand(0.0, 1.0, (3, 4))?;
    let m2 = NdArray::<f32>::rand(0.0, 1.0, (4, 5))?;
    let m3 = m1.matmul(&m2)?;
    
    println!("Result shape: {:?}", m3.shape());
    Ok(())
}
```



## License

MIT



## Reference

- https://github.com/numpy/numpy
- https://github.com/huggingface/candle
