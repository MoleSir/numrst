use crate::{DType, Error, NdArray, Result, Shape, Storage, WithDType};
use std::{fs::File, io::{BufReader, BufWriter, Read, Write}, path::Path};
use derive_builder::Builder;
use bytemuck::{cast_slice, AnyBitPattern};

use super::DynamicNdArray;

#[derive(Debug, Builder)]
#[builder(setter(into))]
pub struct Npy {
    pub descr: String,
    pub version: (u8, u8),
    pub fortran_order: bool,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Npy {
    pub fn from_ndarray<D: WithDType + bytemuck::NoUninit>(ndarray: &NdArray<D>) -> Result<Self> {
        let mut builder = NpyBuilder::default();
        builder.descr(Self::dtype_to_descr(D::DTYPE)?);
        builder.version((1, 0));
        builder.fortran_order(false);
        builder.shape(ndarray.dims());

        let data_vec: Vec<D> = ndarray.storage().copy_data();
        let data_bytes: Vec<u8> = bytemuck::cast_slice(&data_vec).to_vec();
        builder.data(data_bytes);

        builder.build().map_err(|e| Error::Msg(format!("Build npy failed {}", e)))
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
}

impl Npy {
    pub fn to_ndarray(&self) -> Result<DynamicNdArray> {
        match self.descr.as_str() {
            "<f4" => Self::create_ndarray::<f32>(self).map(DynamicNdArray::F32),
            "<f8" => Self::create_ndarray::<f64>(self).map(DynamicNdArray::F64),
            "<i4" => Self::create_ndarray::<i32>(self).map(DynamicNdArray::I32),
            "<u4" => Self::create_ndarray::<u32>(self).map(DynamicNdArray::U32),
            "|b1" => {
                let data: Vec<bool> = self.data.iter().map(|&b| b != 0).collect();
                self.create_ndarray_from_data(data).map(DynamicNdArray::Bool)
            }
            "<u8" => Self::create_ndarray::<usize>(self).map(DynamicNdArray::USize),
            _ => crate::bail!("Unsupported dtype: {}", self.descr),
        }
    }

    fn create_ndarray<D: WithDType + AnyBitPattern>(&self) -> Result<NdArray<D>> {
        let data = cast_slice::<u8, D>(&self.data).to_vec();
        self.create_ndarray_from_data(data)
    }

    fn create_ndarray_from_data<D: WithDType>(&self, data: Vec<D>) -> Result<NdArray<D>> {
        let shape: Shape = self.shape.clone().into();
        if data.len() != shape.element_count() {
            crate::bail!("Unmatch shape '{}' and data len '{}'", shape, data.len())
        }

        let storage = Storage::new(data);
        Ok(NdArray::<D>::from_storage(storage, shape))
    }
}

impl Npy {     
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Npy> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);
        let mut builder = NpyBuilder::default();

        // Read magic
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;
        if &magic != b"\x93NUMPY" {
            return Err(crate::Error::Msg("Not a NPY file".into()));
        }

        // Read version
        let mut version = [0u8; 2];
        reader.read_exact(&mut version)?;
        let version = (version[0], version[1]);
        builder.version(version);

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
                builder.descr(descr);

                header_str = header_str[colon_index+1..].trim_start();
            } else if header_str.starts_with("'fortran_order'") {
                let colon_index = header_str.find(",").ok_or_else(|| Error::Msg("No colon in 'fortran_order' field".into()))?;
                // ": False"
                let fortran_order = &header_str[15..colon_index];
                let index = fortran_order.find(":").ok_or_else(|| Error::Msg("No : in 'fortran_order' field".into()))?;
                // "False"
                let fortran_order = fortran_order[index+1..].trim();
                match fortran_order {
                    "False" => builder.fortran_order(false),
                    "True" => builder.fortran_order(true),
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
                builder.shape(shape);

                header_str = header_str[right_brace_index+2..].trim_start();                
            }
        }

        // Read data
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        builder.data(data);

        Ok(builder.build().map_err(|e| Error::Msg(format!("build fail for '{}'", e)))?)
    }
}

impl Npy {
    pub fn write_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(b"\x93NUMPY")?;

        writer.write_all(&[self.version.0, self.version.1])?;

        let dict = format!(
            "{{'descr': '{}', 'fortran_order': {}, 'shape': ({}), }}",
            self.descr,
            if self.fortran_order { "True" } else { "False" },
            self.shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")
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
        writer.write_all(&self.data)?;

        writer.flush()?;
        Ok(())
    }
}


#[cfg(test)]
mod test {
    use tempfile::NamedTempFile;

    use crate::NdArray;

    use super::Npy;

    #[test]
    fn test_from_ndarray() {
        let ndarray = NdArray::<f32>::randn(0., 1., (4, 5)).unwrap();
        let npy = Npy::from_ndarray(&ndarray).unwrap();
        assert_eq!(npy.shape, [4, 5]);
        assert_eq!(npy.data.len(), 4 * 5 * 4);
    }

    #[test]
    fn test_load_npy() {
        let npy = Npy::load_file("./data/npy/test1.npy").unwrap();
        assert_eq!(npy.shape, [2, 3]);
        assert_eq!(npy.version, (1, 0));
        assert_eq!(npy.descr, "<f4");

        let npy = Npy::load_file("./data/npy/test2.npy").unwrap();
        assert_eq!(npy.shape, [3]);
        assert_eq!(npy.version, (1, 0));
        assert_eq!(npy.descr, "<f8");

        let npy = Npy::load_file("./data/npy/test3.npy").unwrap();
        assert_eq!(npy.shape, [4, 3]);
        assert_eq!(npy.version, (1, 0));
        assert_eq!(npy.descr, "|b1");

        let npy = Npy::load_file("./data/npy/test4.npy").unwrap();
        assert_eq!(npy.shape, [15,]);
        assert_eq!(npy.version, (1, 0));
        assert_eq!(npy.descr, "<i4");

        let npy = Npy::load_file("./data/npy/test5.npy").unwrap();
        assert_eq!(npy.shape, [100, 100]);
        assert_eq!(npy.version, (1, 0));
        assert_eq!(npy.descr, "<f8");
    }

    #[test]
    fn test_to_ndarray() {
        let npy = Npy::load_file("./data/npy/test1.npy").unwrap();
        let ndarray = npy.to_ndarray().unwrap().f32().unwrap();
        println!("{}", ndarray);

        let npy = Npy::load_file("./data/npy/test3.npy").unwrap();
        let ndarray = npy.to_ndarray().unwrap().bool().unwrap();
        println!("{}", ndarray);

        let npy = Npy::load_file("./data/npy/test4.npy").unwrap();
        let ndarray = npy.to_ndarray().unwrap().i32().unwrap();
        println!("{}", ndarray);

        let npy = Npy::load_file("./data/npy/test5.npy").unwrap();
        let _ndarray = npy.to_ndarray().unwrap().f64().unwrap();
    }

    #[test]
    fn test_write_npy() {
        let tmpfile = NamedTempFile::new().unwrap();
        let ndarray = NdArray::<f32>::randn(0., 1., (4, 5)).unwrap();
        let npy = Npy::from_ndarray(&ndarray).unwrap();
        npy.write_file(tmpfile.path()).unwrap();

        let loaded = Npy::load_file(tmpfile.path()).unwrap();
    
        assert_eq!(npy.descr, loaded.descr);
        assert_eq!(npy.shape, loaded.shape);
        assert_eq!(npy.data, loaded.data);

        let loaded_ndarray = npy.to_ndarray().unwrap().f32().unwrap();
        assert!(loaded_ndarray.allclose(&ndarray, 1e-6, 1e-6));
    }
}