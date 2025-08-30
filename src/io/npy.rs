use zip::write::SimpleFileOptions;
use zip::ZipArchive;
use zip::ZipWriter;
use crate::{DType, NdArray, Result, Shape, WithDType};
use super::DynamicNdArray;
use super::utils::*;
use std::collections::HashMap;
use std::io::Cursor;
use std::io::Seek;
use std::{fs::File, io::{BufReader, BufWriter, Read, Write}, path::Path};

/// Load a single `.npy` file into a [`DynamicNdArray`].
///
/// # Arguments
/// * `path` - Path to the `.npy` file.
///
/// # Errors
/// Returns an error if the file cannot be opened, read, or parsed.
pub fn load_npy_file<P: AsRef<Path>>(path: P) -> Result<DynamicNdArray> {
    DynamicNdArray::load_npy_file(path)
}

/// Save a [`DynamicNdArray`] into a `.npy` file.
///
/// # Arguments
/// * `ndarray` - The array to save.
/// * `path` - Path to the output `.npy` file.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn save_npy_file<P: AsRef<std::path::Path>>(ndarray: &DynamicNdArray, path: P) -> Result<()> {
    ndarray.save_npy_file(path)
}

/// Load multiple arrays from a `.npz` archive into a `HashMap`.
///
/// Each entry in the returned map corresponds to one `.npy` file inside the archive,
/// where the key is the file stem (without extension) and the value is the array.
///
/// # Arguments
/// * `path` - Path to the `.npz` archive.
///
/// # Errors
/// Returns an error if the archive cannot be opened, read, or parsed.
pub fn load_npz_file<P: AsRef<Path>>(path: P) -> Result<HashMap<String, DynamicNdArray>> {
    let file = std::fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut arrays = HashMap::new();

    for i in 0..archive.len() {
        let mut zip_file = archive.by_index(i)?;
        let mut buffer = Vec::new();
        use std::io::Read;
        zip_file.read_to_end(&mut buffer)?;
        
        let cursor = Cursor::new(buffer);
        let array = DynamicNdArray::load_npy_reader(cursor)?;

        let name = Path::new(zip_file.name())
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(zip_file.name())
            .to_string();
        arrays.insert(name, array);
    }

    Ok(arrays)
}

/// Save multiple arrays into a `.npz` archive.
///
/// Each array is stored as a separate `.npy` file inside the archive,
/// using the map key as the file name.
///
/// # Arguments
/// * `ndarrays` - A map of name → array pairs to save.
/// * `path` - Path to the output `.npz` archive.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn save_npz_file<P: AsRef<Path>>(ndarrays: &HashMap<String, DynamicNdArray>, path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);

    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .unix_permissions(0o755);
    
    for (name, array) in ndarrays {
        zip.start_file(format!("{}.npy", name), options)?;
        array.save_npy_writer(&mut zip)?;
    }

    zip.finish()?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum NpyError {
    // === Npy Error ===
    #[error("not a npy file for can't read correct magic")]
    NotNpyFile,

    #[error("unsupported NPY version {major}.{minor}")]
    UnsupportVersion {
        major: u8,
        minor: u8,
    },

    #[error("{error} in {field} field")]
    Header {
        error: String,
        field: &'static str 
    },

    #[error("no exit {field} field")]
    LackField {
        field: &'static str 
    },

    #[error("unsupport descr {0}")]
    UnsupportedDescr(String),

    #[error("Numpy does't support usize dtype")]
    UnsupportUSize,
}

const NPY_MAGIC: &[u8] = b"\x93NUMPY";

impl DynamicNdArray {
    pub fn save_npy_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        match self {
            DynamicNdArray::Bool(arr) => arr.save_npy_file(path),
            DynamicNdArray::U8(arr) => arr.save_npy_file(path),
            DynamicNdArray::I8(arr) => arr.save_npy_file(path),
            DynamicNdArray::U16(arr) => arr.save_npy_file(path),
            DynamicNdArray::I16(arr) => arr.save_npy_file(path),
            DynamicNdArray::I32(arr) => arr.save_npy_file(path),
            DynamicNdArray::U32(arr) => arr.save_npy_file(path),
            DynamicNdArray::USize(arr) => arr.save_npy_file(path),
            DynamicNdArray::F32(arr) => arr.save_npy_file(path),
            DynamicNdArray::F64(arr) => arr.save_npy_file(path),
        }
    }

    pub fn save_npy_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        match self {
            DynamicNdArray::Bool(arr) => arr.save_npy_writer(writer),
            DynamicNdArray::U8(arr) => arr.save_npy_writer(writer),
            DynamicNdArray::I8(arr) => arr.save_npy_writer(writer),
            DynamicNdArray::U16(arr) => arr.save_npy_writer(writer),
            DynamicNdArray::I16(arr) => arr.save_npy_writer(writer),
            DynamicNdArray::I32(arr) => arr.save_npy_writer(writer),
            DynamicNdArray::U32(arr) => arr.save_npy_writer(writer),
            DynamicNdArray::USize(arr) => arr.save_npy_writer(writer),
            DynamicNdArray::F32(arr) => arr.save_npy_writer(writer),
            DynamicNdArray::F64(arr) => arr.save_npy_writer(writer),
        }
    }
}

impl<D: WithDType + bytemuck::NoUninit> NdArray<D> {
    pub fn save_npy_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        let descr = dtype_to_descr(D::DTYPE)?;
        let version = (1, 0);
        let fortran_order = false;
        let shape = self.dims();

        let storage = self.storage();
        let data_vec = storage.data();
        let bytes: Vec<u8> = bytemuck::cast_slice(data_vec).to_vec();

        writer.write_all(NPY_MAGIC)?;

        writer.write_all(&[version.0, version.1])?;

        let dict = format!(
            "{{'descr': '{}', 'fortran_order': {}, 'shape': ({}), }}",
            descr,
            if fortran_order { "True" } else { "False" },
            shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")
        );

        // header padding 
        let header_len = dict.len() + 1; // \n
        if header_len > u16::MAX as usize {
            panic!("Header too long for version 1.0 NPY");
        }
        let padding = (64 - ((10 + 2 + header_len) % 64)) % 64;
        let mut header_bytes = dict.into_bytes();
        header_bytes.push(b'\n');
        header_bytes.extend(vec![b' '; padding]);

        // header_len
        let header_len_u16: u16 = header_bytes.len().try_into().unwrap();
        writer.write_all(&header_len_u16.to_le_bytes())?;

        // header
        writer.write_all(&header_bytes)?;

        // data
        writer.write_all(&bytes)?;

        writer.flush()?;
        Ok(())
    }

    pub fn save_npy_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        // Write file
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.save_npy_writer(&mut writer)
    }
}

impl DynamicNdArray {
    pub fn load_npy_reader<R: Read + Seek>(mut reader: R) -> Result<Self> {
        // Read magic
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;
        if &magic != NPY_MAGIC {
            Err( NpyError::NotNpyFile )?;
        }

        // Read version
        let mut version = [0u8; 2];
        reader.read_exact(&mut version)?;
        let version = (version[0], version[1]);

        // Read header
        let header_len = match version {
            (1, 0) => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)?;
                u16::from_le_bytes(buf) as usize
            }
            (2, 0) | (3, 0) => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                u32::from_le_bytes(buf) as usize
            }
            _ => Err( NpyError::UnsupportVersion { major: version.0, minor: version.1 })?,
        };

        // Read Header
        let mut header_bytes = vec![0u8; header_len];   
        reader.read_exact(&mut header_bytes)?;
        let header_str = str::from_utf8(&header_bytes)?.trim();

        // Parse header
        // "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }"
        let header_str = header_str.trim_matches(|c| c == '{' || c == '}');

        // 'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), 
        let mut descr_opt: Option<&str> = None;
        let mut fortran_order_opt: Option<bool> = None;
        let mut shape_opt: Option<Vec<usize>> = None;

        let mut header_str = header_str;
        while !header_str.is_empty() {
            if header_str.starts_with("'descr'") {
                let colon_index = header_str.find(",")
                    .ok_or_else(|| NpyError::Header { 
                        field: "descr", 
                        error: "No colon".into() 
                    })?;
                // ": '<f8'"
                let descr = &header_str[7..colon_index];
                let ref1_index = descr.find("'")
                    .ok_or_else(|| NpyError::Header { 
                        field: "descr", 
                        error: "No start ' in 'descr' field value".into() 
                    })?;
                // "<f8'"
                let descr = &descr[ref1_index+1..];
                let ref2_index = descr.find("'")
                    .ok_or_else(|| NpyError::Header { 
                        field: "descr", 
                        error: "No end ' in 'descr' field value".into() 
                    })?;
                // "<f8"
                let descr = &descr[..ref2_index];
                descr_opt = Some(descr);

                header_str = header_str[colon_index+1..].trim_start();
            } else if header_str.starts_with("'fortran_order'") {
                let colon_index = header_str.find(",")
                    .ok_or_else(|| NpyError::Header { 
                        field: "fortran_order", 
                        error: "No colon in 'fortran_order' field".into() 
                    })?;
                // ": False"
                let fortran_order = &header_str[15..colon_index];
                let index = fortran_order.find(":")
                    .ok_or_else(|| NpyError::Header { 
                        field: "fortran_order", 
                        error: "No : in 'fortran_order' field".into() 
                    })?;
                // "False"
                let fortran_order = fortran_order[index+1..].trim();
                match fortran_order {
                    "False" => fortran_order_opt = Some(false),
                    "True" => fortran_order_opt = Some(true),
                    _ => Err(NpyError::Header { 
                        field: "fortran_order", 
                        error: format!("Unsupported value '{}'", fortran_order) 
                    })?,
                };

                header_str = header_str[colon_index+1..].trim_start();
            } else if header_str.starts_with("'shape'") {
                // "'shape': (3, 4), "
                let left_brace_index = header_str.find("(")
                    .ok_or_else(|| NpyError::Header { 
                        field: "shape", 
                        error: "No ( in 'shape' field".into() 
                    })?;
                let right_brace_index = header_str.find(")")
                    .ok_or_else(|| NpyError::Header { 
                        field: "shape", 
                        error: "No ) in 'shape' field".into() 
                    })?;
                // 3, 4
                let shape = &header_str[left_brace_index + 1..right_brace_index];
                let shape: Vec<usize> = shape
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect();
                shape_opt = Some(shape);

                header_str = header_str[right_brace_index+2..].trim_start();                
            }
        }

        // Check header
        let descr = descr_opt.ok_or_else(|| NpyError::LackField { field: "descr" } )?;
        let _ = fortran_order_opt.ok_or_else(|| NpyError::LackField { field: "fortran_order" } )?;
        let shape = shape_opt.ok_or_else(|| NpyError::LackField { field: "shape" } )?;
        let shape: Shape = shape.into();

        match descr {
            "|b1" => read_bool_ndarray::<R>(reader, shape).map(DynamicNdArray::Bool),
            "<i1" => read_ndarray::<i8, R>(reader, shape).map(DynamicNdArray::I8),
            "<u1" => read_ndarray::<u8, R>(reader, shape).map(DynamicNdArray::U8),
            "<i2" => read_ndarray::<i16, R>(reader, shape).map(DynamicNdArray::I16),
            "<u2" => read_ndarray::<u16, R>(reader, shape).map(DynamicNdArray::U16),
            "<i4" => read_ndarray::<i32, R>(reader, shape).map(DynamicNdArray::I32),
            "<u4" => read_ndarray::<u32, R>(reader, shape).map(DynamicNdArray::U32),
            "<f4" => read_ndarray::<f32, R>(reader, shape).map(DynamicNdArray::F32),
            "<f8" => read_ndarray::<f64, R>(reader, shape).map(DynamicNdArray::F64),
            "<u8" => read_ndarray::<usize, R>(reader, shape).map(DynamicNdArray::USize),
            _ => Err(NpyError::UnsupportedDescr(descr.to_string()))?
        }
    }

    pub fn load_npy_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        Self::load_npy_reader(reader)
    }
}

fn dtype_to_descr(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::Bool => Ok("|b1"),
        DType::I8 => Ok("<i1"),
        DType::U8 => Ok("<u1"),
        DType::U16 => Ok("<u2"),
        DType::I16 => Ok("<i2"),
        DType::U32 => Ok("<u4"),
        DType::I32 => Ok("<i4"),
        DType::USize => Err(NpyError::UnsupportUSize)?,
        DType::F32 => Ok("<f4"),
        DType::F64 => Ok("<f8"),
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use tempfile::NamedTempFile;
    use crate::{io::DynamicNdArray, NdArray};

    use super::{load_npz_file, save_npz_file};

    #[test]
    fn test_to_ndarray() {
        let ndarray = DynamicNdArray::load_npy_file("./data/npy/test1.npy").unwrap().f32().unwrap();
        println!("{}", ndarray);

        let ndarray = DynamicNdArray::load_npy_file("./data/npy/test3.npy").unwrap().bool().unwrap();
        println!("{}", ndarray);

        let ndarray = DynamicNdArray::load_npy_file("./data/npy/test4.npy").unwrap().i32().unwrap();
        println!("{}", ndarray);

        let _ndarray = DynamicNdArray::load_npy_file("./data/npy/test5.npy").unwrap().f64().unwrap();
    }

    #[test]
    fn test_write_npy() {
        let tmpfile = NamedTempFile::new().unwrap();
        let ndarray = NdArray::<f32>::randn(0., 1., (4, 5)).unwrap();
        ndarray.save_file(tmpfile.path()).unwrap();

        let loaded_ndarray = DynamicNdArray::load_file(tmpfile.path()).unwrap().f32().unwrap();
        assert!(loaded_ndarray.allclose(&ndarray, 1e-6, 1e-6));
    }

    #[test]
    fn test_load_npz() {
        let ndarrays = load_npz_file("./data/npy/test1.npz").unwrap();
        for (name, _) in ndarrays {
            println!("{}", name);
        }
    }

    #[test]
    fn test_save_npz() {
        let tmpfile = NamedTempFile::new().unwrap();

        let scalar = NdArray::new(1).unwrap();
        let vector_f32 = NdArray::new(&[1.0f32, 2., 3.]).unwrap();
        let matrix_f32 = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let ones_f32 = NdArray::<f32>::ones((2, 9)).unwrap();
        let randn_f64 = NdArray::randn(0.0f64, 1., (1, 2, 3)).unwrap();
        let fill_f64 = NdArray::fill((2, 3, 4), 1.2).unwrap();
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

        save_npz_file(&ndarrays, tmpfile.path()).unwrap();

        let ndarrays = load_npz_file(tmpfile.path()).unwrap();
        for (name, _) in ndarrays {
            println!("{}", name);
        }
    }
}