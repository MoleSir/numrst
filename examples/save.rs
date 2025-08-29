use std::fs::File;
use std::io::BufReader;
use zip::ZipArchive;

fn main() -> zip::result::ZipResult<()> {
    let file = File::open("example.npz")?;
    let reader = BufReader::new(file);
    let mut archive = ZipArchive::new(reader)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        println!("File {}: {}", i, file.name());
        // 这里可以把 file 的内容读取到 Vec<u8>
        let mut data = Vec::new();
        use std::io::Read;
        file.read_to_end(&mut data)?;
        // data 就是 .npy 文件的原始 bytes
    }

    Ok(())
}
