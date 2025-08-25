# NumRst

NumRst is the fundamental package for scientific computing with Rust.



## Features

- [x] create `NdArray` method
- [x] basic binary and unary op
- [x] matmul 
- [x] index op like numpy
- [x] reduce op 
- [x] simple display 



## Examples

Create `NdArray`

```rust
  let arr = NdArray::new(1)?;
  let arr = NdArray::new(&[1.0f32, 2., 3.])?;
  let arr = NdArray::new(&[[1, 2, 3], [3, 4, 5]])?;
  let arr = NdArray::<f32>::ones(1)?;
  let arr = NdArray::randn(0.0f64, 1., (1, 2, 3))?;
  let arr = NdArray::fill((2, 3, 4), 1.2)?;
  let arr = NdArray::fill((10, 4, 4), 100u32)?;
  let arr = NdArray::arange(0., 10.)?;
  let ts = NdArray::trues((3, 4))?;
```

Some basic op

```rust
  let a = NdArray::new(&[1.0f32, 2., 3.])?;
  let b = NdArray::new(&[1.0f32, 2., 3.])?;
  let c = a.add(&b)?;
  let c = a.sub(&b)?;

  let a = NdArray::new(&[[1, 2, 3], [4, 5, 6]])?;
  let sum = a.sum_axis(1)?;

  let a = NdArray::<f32>::randn(0., 1., (4, 4, 5))?;
  let b = NdArray::<f32>::randn(0., 1., (4, 5, 3))?;
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
