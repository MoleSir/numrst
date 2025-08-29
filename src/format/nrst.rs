use std::{fs::File, io::{BufReader, BufWriter, Read, Write}, path::Path};
use crate::{DType, Error, NdArray, Result, Shape, Storage, WithDType};

use super::DynamicNdArray;
const NRST_MAGIC: &[u8] = b"\x25NUMRST";

pub struct NRst {
    pub shape: Shape,
    pub dtype: DType,
    pub data: Vec<u8>,
}


/*
    | MAGIC(7bytes) | DTYPE(1byte) | RANK(4bytes) | SHAPE(RANK * 4bytes) | DATA(..) |
*/

impl DynamicNdArray {
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut reader: BufReader<File> = BufReader::new(file);

        // Read magic
        let mut magic = [0u8; 7];
        reader.read_exact(&mut magic)?;
        if &magic != NRST_MAGIC {
            return Err(crate::Error::Msg("Not a NRST file".into()));
        }

        let mut dtype = [0u8];
        reader.read_exact(&mut dtype)?;
        let dtype = decode_dtype(dtype[0])?;

        let mut rank = [0u8; 4];
        reader.read_exact(&mut rank)?;
        let rank = u32::from_le_bytes(rank);

        let mut shape = vec![]; 
        for _ in 0..rank { 
            let mut bytes = [0u8; 4]; 
            reader.read_exact(&mut bytes)?; 
            shape.push(u32::from_le_bytes(bytes) as usize); 
        }
        let shape: Shape = shape.into();
        assert_eq!(shape.rank(), rank as usize);

        match dtype {
            DType::Bool => read_bool_ndarray(reader, shape).map(DynamicNdArray::Bool),
            DType::I32 => read_ndarray::<i32>(reader, shape).map(DynamicNdArray::I32),
            DType::U32 => read_ndarray::<u32>(reader, shape).map(DynamicNdArray::U32),
            DType::USize => read_ndarray::<usize>(reader, shape).map(DynamicNdArray::USize),
            DType::F32 => read_ndarray::<f32>(reader, shape).map(DynamicNdArray::F32),
            DType::F64 => read_ndarray::<f64>(reader, shape).map(DynamicNdArray::F64),

        }
    }
}

impl<W: WithDType + bytemuck::NoUninit> NdArray<W> {
    pub fn save_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // MAGIC(7bytes)
        writer.write_all(NRST_MAGIC)?;
        // DTYPE(1byte)
        writer.write_all(&[encode_dtype(W::DTYPE)])?;
        // RANK(4bytes)
        let rank: u32 = self.rank() as u32;
        writer.write_all(&rank.to_le_bytes())?;

        // SHAPE(RANK * 4bytes)
        assert!(self.rank() == self.dims().len());
        for dim in self.dims() {
            let dim = *dim as u32;
            writer.write_all(&dim.to_le_bytes())?;
        }
        
        // DATA(..)
        let storage = self.storage();
        let data_vec = storage.data();
        assert!(data_vec.len() == self.element_count());
        let data_bytes: Vec<u8> = bytemuck::cast_slice(data_vec).to_vec();
        writer.write_all(&data_bytes)?;

        Ok(())
    }
}

fn read_ndarray<W: WithDType + bytemuck::Pod>(mut reader: BufReader<File>, shape: Shape) -> Result<NdArray<W>> {
    let element_size = shape.element_count();
    let data = read_as_vec::<W, _>(&mut reader, element_size)?;

    let storage = Storage::new(data);
    Ok(NdArray::from_storage(storage, shape))
}

fn read_bool_ndarray(mut reader: BufReader<File>, shape: Shape) -> Result<NdArray<bool>> {
    let element_size = shape.element_count();
    let data = read_bools(&mut reader, element_size)?;

    let storage = Storage::new(data);
    Ok(NdArray::from_storage(storage, shape))
}

fn read_bools<R: Read>(reader: &mut R, len: usize) -> Result<Vec<bool>> {
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let vec = buf.into_iter().map(|b| b != 0).collect();
    Ok(vec)
}

fn read_as_vec<T: bytemuck::Pod, R: Read>(reader: &mut R, len: usize) -> Result<Vec<T>> {
    let mut buf = vec![0u8; len * std::mem::size_of::<T>()];
    reader.read_exact(&mut buf)?;
    let slice: &[T] = bytemuck::cast_slice(&buf);
    Ok(slice.to_vec())
}

fn encode_dtype(dtype: DType) -> u8 {
    match dtype {
        DType::Bool      => 1,
        DType::U32       => 6,
        DType::I32       => 7,
        DType::F32       => 11,
        DType::F64       => 12,
        DType::USize     => 13,
    }
}

fn decode_dtype(code: u8) -> Result<DType> {
    match code {
        1  => Ok(DType::Bool),
        6  => Ok(DType::U32),
        7  => Ok(DType::I32),
        11 => Ok(DType::F32),
        12 => Ok(DType::F64),
        13 => Ok(DType::USize),
        _  => Err(Error::Msg(format!("Unknown dtype code: {}", code))),
    }
}


#[cfg(test)]
mod test {
    use tempfile::NamedTempFile;

    use crate::{format::DynamicNdArray, NdArray};

    #[test]
    pub fn test_save() {
        let tmpfile = NamedTempFile::new().unwrap();

        let ndarray = NdArray::<f32>::randn(0., 1., (4, 5, 9)).unwrap();
        ndarray.save_file(tmpfile.path()).unwrap();
    }

    #[test]
    pub fn test_load() {
        let arr = DynamicNdArray::load_file("./data/nrst/test1.nrst").unwrap();
        let arr = arr.f32().unwrap();
        assert_eq!(arr.dims(), [4, 5, 9]);
    }

    #[test]
    pub fn test_save_load() {
        let tmpfile = NamedTempFile::new().unwrap();

        let arr = NdArray::<f64>::rand(-10., 10., (3, 5, 7)).unwrap();
        arr.save_file(tmpfile.path()).unwrap();
        let arr_loaded = DynamicNdArray::load_file(tmpfile.path()).unwrap().f64().unwrap();

        assert!(arr.allclose(&arr_loaded, 1e-4, 1e-4))
    }
}