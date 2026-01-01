use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MPV_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MPV_PREFIX");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_STT_REMOTE_UDP");

    if env::var("CARGO_FEATURE_STT_REMOTE_UDP").is_ok() {
        println!("cargo:rerun-if-changed=src/opusenc_shim.c");
        cc::Build::new()
            .file("src/opusenc_shim.c")
            .compile("opusenc_shim");
    }

    let target = env::var("TARGET").unwrap_or_default();
    if !target.contains("android") {
        return;
    }

    if let Ok(prefix) = env::var("MPV_PREFIX") {
        let lib_dir = PathBuf::from(prefix).join("lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    } else if let Ok(lib_dir) = env::var("MPV_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    }

    println!("cargo:rustc-link-lib=mpv");
}
