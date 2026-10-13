use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=kernels/fluid.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("fluid.ptx");
    let cap = env::var("CUDA_COMPUTE_CAP").ok().unwrap_or_else(|| {
        Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.lines().next().map(|l| l.trim().replace('.', "")))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "80".to_string())
    });
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let status = Command::new(&nvcc)
        .args(["--ptx", "-O3", "--use_fast_math", "-std=c++17"])
        .arg(format!("-arch=compute_{cap}"))
        .arg("kernels/fluid.cu")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("failed to run nvcc (set NVCC or put it on PATH)");
    assert!(status.success(), "nvcc failed");
}
