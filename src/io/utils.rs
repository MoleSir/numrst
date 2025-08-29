use std::io::Read;
use crate::{NdArray, Result, Shape, Storage, WithDType};

pub fn read_ndarray<W: WithDType + bytemuck::Pod, R: Read>(mut reader: R, shape: Shape) -> Result<NdArray<W>> {
    let element_size = shape.element_count();
    let data = read_as_vec::<W, _>(&mut reader, element_size)?;

    let storage = Storage::new(data);
    Ok(NdArray::from_storage(storage, shape))
}

pub fn read_bool_ndarray<R: Read>(mut reader: R, shape: Shape) -> Result<NdArray<bool>> {
    let element_size = shape.element_count();
    let data = read_bools(&mut reader, element_size)?;

    let storage = Storage::new(data);
    Ok(NdArray::from_storage(storage, shape))
}

pub fn read_bools<R: Read>(reader: &mut R, len: usize) -> Result<Vec<bool>> {
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let vec = buf.into_iter().map(|b| b != 0).collect();
    Ok(vec)
}

pub fn read_as_vec<T: bytemuck::Pod, R: Read>(reader: &mut R, len: usize) -> Result<Vec<T>> {
    let mut buf = vec![0u8; len * std::mem::size_of::<T>()];
    reader.read_exact(&mut buf)?;
    let slice: &[T] = bytemuck::cast_slice(&buf);
    Ok(slice.to_vec())
}