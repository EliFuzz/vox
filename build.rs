use std::env::var;

fn main() {
    println!("cargo:rerun-if-changed=vendor/cactus/libneedle.a");
    let is_apple_silicon = var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64");
    if !is_apple_silicon {
        return;
    }
    let manifest_dir = var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-search=native={manifest_dir}/vendor/cactus");
    println!("cargo:rustc-link-lib=static=needle");
    println!("cargo:rustc-link-lib=c++");
}
