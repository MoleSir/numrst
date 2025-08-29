use crate::{DType, Error, NdArray, Result, Shape, WithDType};
use super::DynamicNdArray;
use super::utils::*;
use std::{fs::File, io::{BufReader, BufWriter, Read, Write}, path::Path};

const NPY_MAGIC: &[u8] = b"\x93NUMPY";

impl<D: WithDType + bytemuck::NoUninit> NdArray<D> {
    pub fn save_npy_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        // Infomation
        let descr = dtype_to_descr(D::DTYPE)?;
        let version = (1, 0);
        let fortran_order = false;
        let shape = self.dims();

        let storage = self.storage();
        let data_vec = storage.data();
        let bytes: Vec<u8> = bytemuck::cast_slice(data_vec).to_vec();

        // Write file
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

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
}

impl DynamicNdArray {
    pub fn load_npy_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Read magic
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;
        if &magic != NPY_MAGIC {
            return Err(crate::Error::Msg("Not a NPY file".into()));
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
            _ => return Err(crate::Error::Msg(format!("Unsupported NPY version {}.{}", version.0, version.1).into())),
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
                let colon_index = header_str.find(",").ok_or_else(|| Error::Msg("No colon in 'descr' field".into()))?;
                // ": '<f8'"
                let descr = &header_str[7..colon_index];
                let ref1_index = descr.find("'").ok_or_else(|| Error::Msg("No start ' in 'descr' field value".into()))?;
                // "<f8'"
                let descr = &descr[ref1_index+1..];
                let ref2_index = descr.find("'").ok_or_else(|| Error::Msg("No end ' in 'descr' field value".into()))?;
                // "<f8"
                let descr = &descr[..ref2_index];
                descr_opt = Some(descr);

                header_str = header_str[colon_index+1..].trim_start();
            } else if header_str.starts_with("'fortran_order'") {
                let colon_index = header_str.find(",").ok_or_else(|| Error::Msg("No colon in 'fortran_order' field".into()))?;
                // ": False"
                let fortran_order = &header_str[15..colon_index];
                let index = fortran_order.find(":").ok_or_else(|| Error::Msg("No : in 'fortran_order' field".into()))?;
                // "False"
                let fortran_order = fortran_order[index+1..].trim();
                match fortran_order {
                    "False" => fortran_order_opt = Some(false),
                    "True" => fortran_order_opt = Some(true),
                    _ => Err(Error::Msg(format!("Unsupport fortran_order '{}'", fortran_order)))?,
                };

                header_str = header_str[colon_index+1..].trim_start();
            } else if header_str.starts_with("'shape'") {
                // "'shape': (3, 4), "
                let left_brace_index = header_str.find("(").ok_or_else(|| Error::Msg("No ( in 'shape' field".into()))?;
                let right_brace_index = header_str.find(")").ok_or_else(|| Error::Msg("No ) in 'shape' field".into()))?;
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
        let descr = descr_opt.ok_or_else(|| Error::Msg(format!("Un descr infomtion")))?;
        let _ = fortran_order_opt.ok_or_else(|| Error::Msg(format!("Un fortran_order infomtion")))?;
        let shape = shape_opt.ok_or_else(|| Error::Msg(format!("Un shape infomtion")))?;
        let shape: Shape = shape.into();

        match descr {
            "<f4" => read_ndarray::<f32>(reader, shape).map(DynamicNdArray::F32),
            "<f8" => read_ndarray::<f64>(reader, shape).map(DynamicNdArray::F64),
            "<i4" => read_ndarray::<i32>(reader, shape).map(DynamicNdArray::I32),
            "<u4" => read_ndarray::<u32>(reader, shape).map(DynamicNdArray::U32),
            "|b1" => read_bool_ndarray(reader, shape).map(DynamicNdArray::Bool),
            "<u8" => read_ndarray::<usize>(reader, shape).map(DynamicNdArray::USize),
            _ => crate::bail!("Unsupported dtype: {}", descr),
        }
    }
}

fn dtype_to_descr(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::Bool => Ok("|b1"),
        DType::U32 => Ok("<u4"),
        DType::I32 => Ok("<i4"),
        DType::USize => crate::bail!("Numpy does't support usize dtype"),
        DType::F32 => Ok("<f4"),
        DType::F64 => Ok("<f8"),
    }
}

#[cfg(test)]
mod test {
    use tempfile::NamedTempFile;
    use crate::{format::DynamicNdArray, NdArray};

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
}