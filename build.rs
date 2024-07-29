use std::{
    env,
    error::Error,
    fs::{self, File},
    path::PathBuf,
};

fn main() -> Result<(), Box<dyn Error>> {
    let source_files = &["cuda/layer/layer.cu", "cuda/layer/dense.cu", "cuda/lib.cu"];
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        .includes(&["cuda"])
        .files(source_files)
        .compile("ai");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    for header_file in &["cuda", "cuda/layer"] {
        println!("cargo:rerun-if-changed={}", header_file);
    }

    Ok(())
}
