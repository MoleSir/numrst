# NumRs

NumRs is the fundamental package for scientific computing with Rust.



## Features

- [x] create `NdArray` method
- [x] basic binary and unary op
- [x] matmul 
- [x] index op like numpy
- [x] display 



## Examples

Create `NdArray`

```rust
let arr = NdArray::new(1)?;
let arr = NdArray::new(&[1.0f32, 2., 3.])?;
let arr = NdArray::ones(1, DType::I32)?;
let arr = NdArray::randn(0., 1., (1, 2, 3), DType::F64)?;
let arr = NdArray::arange(0., 10.)?;
```

Some basic op

```rust
let a = NdArray::new(&[1.0f32, 2.0, 3.0])?;
let b = NdArray::new(&[4.0f32, 5.0, 6.0])?;
let c = a.add(&b)?;
let c = a.sub(&b)?;
let c = a.mul(&b)?;
let c = a.div(&b)?;
let c = a.exp()?;
let c = a.relu()?;

let a = NdArray::arange(0., 12.).unwrap().reshape((2, 2, 3))?;
let b = NdArray::arange(0., 12.).unwrap().reshape((2, 3, 2))?;
let c = a.matmul(&b)?;
```

Usefull index op like NumPy

```rust
let arr = NdArray::zeros((5, 5, 5), DType::U32)?;

let sub_arr = arr.index(1)?;
let sub_arr = arr.index(2)?;
let sub_arr = arr.index(rng!(1:3))?;
let sub_arr = arr.index((rng!(1:3), rng!(3:4), 1))?;
let sub_arr = arr.index((rng!(1:3), .., 1..2))?;
```



## LICENSE

MIT



## References

- https://github.com/numpy/numpy
- https://github.com/huggingface/candle
