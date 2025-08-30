use std::collections::HashMap;

use numrst::{io::DynamicNdArray, prelude::*};
use tempfile::NamedTempFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    numrst::io::save_npz_file(&ndarrays, tmpfile.path()).unwrap();

    let ndarrays = numrst::io::load_npz_file(tmpfile.path()).unwrap();
    for (name, _) in ndarrays {
        println!("{}", name);
    }

    Ok(())
}
