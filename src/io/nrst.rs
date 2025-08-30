use zip::{write::SimpleFileOptions, ZipArchive, ZipWriter};

use super::utils::*;
use crate::{DType, NdArray, Result, Shape, WithDType};
use super::DynamicNdArray;

use std::{collections::HashMap, fs::File, io::{BufReader, BufWriter, Cursor, Read, Seek, Write}, path::Path};

/// Load a single `.nrst` file from disk into a `DynamicNdArray`.
///
/// # Arguments
/// * `path` - Path to the `.npy` file.
///
/// # Returns
/// A `DynamicNdArray` containing the data from the file.
pub fn load_file<P: AsRef<Path>>(path: P) -> Result<DynamicNdArray> {
    DynamicNdArray::load_file(path)
}

/// Save a single `DynamicNdArray` to a `.nrst` file on disk.
///
/// # Arguments
/// * `ndarray` - Reference to the `DynamicNdArray` to save.
/// * `path` - Path where the `.nrst` file will be written.
///
/// # Returns
/// An empty `Result` indicating success or failure.
pub fn save_file<P: AsRef<Path>>(ndarray: &DynamicNdArray, path: P) -> Result<()> {
    ndarray.save_file(path)
}

/// Load a `.nrsz` archive file (ZIP format containing multiple `.nrst` files).
///
/// # Arguments
/// * `path` - Path to the `.nrsz` file.
///
/// # Returns
/// A `HashMap` where each entry maps the array name (filename without extension)
/// to its corresponding `DynamicNdArray`.
pub fn load_zfile<P: AsRef<Path>>(path: P) -> Result<HashMap<String, DynamicNdArray>> {
    let file = std::fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut arrays = HashMap::new();

    for i in 0..archive.len() {
        let mut zip_file = archive.by_index(i)?;
        let mut buffer = Vec::new();
        use std::io::Read;
        zip_file.read_to_end(&mut buffer)?;
        
        let cursor = Cursor::new(buffer);
        let array = DynamicNdArray::load_reader(cursor)?;
        let name = Path::new(zip_file.name())
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(zip_file.name())
            .to_string();

        arrays.insert(name, array);
    }

    Ok(arrays)
}

/// Save multiple `DynamicNdArray`s into a `.nrsz` archive (ZIP format).
///
/// Each array is stored as a separate `.npy` file inside the archive.
/// The `HashMap` keys are used as the file names (without extension).
///
/// # Arguments
/// * `ndarrays` - A `HashMap` mapping names to `DynamicNdArray`s.
/// * `path` - Path where the `.nrsz` file will be created.
///
/// # Returns
/// An empty `Result` indicating success or failure.
pub fn save_zfile<P: AsRef<Path>>(ndarrays: &HashMap<String, DynamicNdArray>, path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);

    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .unix_permissions(0o755);
    
    for (name, array) in ndarrays {
        zip.start_file(format!("{}.npy", name), options)?;
        array.save_writer(&mut zip)?;
    }

    zip.finish()?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum NrstError {
    // === Npy Error ===
    #[error("not a npy file for can't read correct magic")]
    NotNrstFile,

    #[error("Invalid dtype code: {0}")]
    InvalidDTypeCode(u8),
}

const NRST_MAGIC: &[u8] = b"\x25NUMRST";

/*
    | MAGIC(7bytes) | DTYPE(1byte) | RANK(4bytes) | SHAPE(RANK * 4bytes) | DATA(..) |
*/

impl DynamicNdArray {
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader: BufReader<File> = BufReader::new(file);
        Self::load_reader(reader)
    }

    pub fn load_reader<R: Read + Seek>(mut reader: R) -> Result<Self> {
        // Read magic
        let mut magic = [0u8; 7];
        reader.read_exact(&mut magic)?;
        if &magic != NRST_MAGIC {
            Err(NrstError::NotNrstFile)?;
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
            DType::U8 => read_ndarray::<u8, R>(reader, shape).map(DynamicNdArray::U8),
            DType::I8 => read_ndarray::<i8, R>(reader, shape).map(DynamicNdArray::I8),
            DType::I16 => read_ndarray::<i16, R>(reader, shape).map(DynamicNdArray::I16),
            DType::U16 => read_ndarray::<u16, R>(reader, shape).map(DynamicNdArray::U16),
            DType::I32 => read_ndarray::<i32, R>(reader, shape).map(DynamicNdArray::I32),
            DType::U32 => read_ndarray::<u32, R>(reader, shape).map(DynamicNdArray::U32),
            DType::USize => read_ndarray::<usize, R>(reader, shape).map(DynamicNdArray::USize),
            DType::F32 => read_ndarray::<f32, R>(reader, shape).map(DynamicNdArray::F32),
            DType::F64 => read_ndarray::<f64, R>(reader, shape).map(DynamicNdArray::F64),

        }
    }

    pub fn save_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        match self {
            DynamicNdArray::Bool(arr) => arr.save_file(path),
            DynamicNdArray::U8(arr) => arr.save_file(path),
            DynamicNdArray::I8(arr) => arr.save_file(path),
            DynamicNdArray::U16(arr) => arr.save_file(path),
            DynamicNdArray::I16(arr) => arr.save_file(path),
            DynamicNdArray::I32(arr) => arr.save_file(path),
            DynamicNdArray::U32(arr) => arr.save_file(path),
            DynamicNdArray::USize(arr) => arr.save_file(path),
            DynamicNdArray::F32(arr) => arr.save_file(path),
            DynamicNdArray::F64(arr) => arr.save_file(path),
        }
    }

    pub fn save_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        match self {
            DynamicNdArray::Bool(arr) => arr.save_writer(writer),
            DynamicNdArray::U8(arr) => arr.save_writer(writer),
            DynamicNdArray::I8(arr) => arr.save_writer(writer),
            DynamicNdArray::U16(arr) => arr.save_writer(writer),
            DynamicNdArray::I16(arr) => arr.save_writer(writer),
            DynamicNdArray::I32(arr) => arr.save_writer(writer),
            DynamicNdArray::U32(arr) => arr.save_writer(writer),
            DynamicNdArray::USize(arr) => arr.save_writer(writer),
            DynamicNdArray::F32(arr) => arr.save_writer(writer),
            DynamicNdArray::F64(arr) => arr.save_writer(writer),
        }
    }
}

impl<T: WithDType + bytemuck::NoUninit> NdArray<T> {
    pub fn save_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.save_writer(&mut writer)
    }

    pub fn save_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        // MAGIC(7bytes)
        writer.write_all(NRST_MAGIC)?;
        // DTYPE(1byte)
        writer.write_all(&[encode_dtype(T::DTYPE)])?;
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

fn encode_dtype(dtype: DType) -> u8 {
    match dtype {
        DType::Bool      => 1,
        DType::U8        => 2,
        DType::I8        => 3,
        DType::U16       => 4,
        DType::I16       => 5,
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
        2  => Ok(DType::U8),
        3  => Ok(DType::I8),
        4  => Ok(DType::U16),
        5  => Ok(DType::I16),
        6  => Ok(DType::U32),
        7  => Ok(DType::I32),
        11 => Ok(DType::F32),
        12 => Ok(DType::F64),
        13 => Ok(DType::USize),
        _  => Err(NrstError::InvalidDTypeCode(code))?,
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use tempfile::NamedTempFile;
    use crate::{io::DynamicNdArray, NdArray};
    use super::{load_zfile, save_zfile};

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

    #[test]
    fn test_save_npz() {
        let tmpfile = NamedTempFile::new().unwrap();

        let scalar = NdArray::new(1).unwrap();
        let vector_f32 = NdArray::new(&[1.0f32, 2., 3.]).unwrap();
        let matrix_f32 = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let ones_f32 = NdArray::<f32>::ones((2, 9)).unwrap();
        let randn_f64 = NdArray::randn(0.0f64, 1., (1, 2, 3)).unwrap();
        let fill_f64 = NdArray::full((2, 3, 4), 1.2).unwrap();
        let arange_f64 = NdArray::arange(0., 10.).unwrap();
        let trues = NdArray::trues((3, 4)).unwrap();
        let booleans = NdArray::new(&[[true, false], [false, true]]).unwrap();


        let mut ndarrays = HashMap::new();
        ndarrays.insert("scalar".to_string(), DynamicNdArray::I32(scalar));
        ndarrays.insert("vector_f32".to_string(), DynamicNdArray::F32(vector_f32));
        ndarrays.insert("matrix_f32".to_string(), DynamicNdArray::I32(matrix_f32));
        ndarrays.insert("ones_f32".to_string(), DynamicNdArray::F32(ones_f32));
        ndarrays.insert("randn_f64".to_string(), DynamicNdArray::F64(randn_f64));
        ndarrays.insert("fill_f64".to_string(), DynamicNdArray::F64(fill_f64));
        ndarrays.insert("arange_f64".to_string(), DynamicNdArray::F64(arange_f64));
        ndarrays.insert("trues".to_string(), DynamicNdArray::Bool(trues));
        ndarrays.insert("booleans".to_string(), DynamicNdArray::Bool(booleans));

        save_zfile(&ndarrays, tmpfile.path()).unwrap();

        let ndarrays = load_zfile(tmpfile.path()).unwrap();
        for (name, _) in ndarrays {
            println!("{}", name);
        }
    }
}