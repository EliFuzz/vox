#[cfg(target_os = "linux")]
pub const PLATFORM_NAME: &str = "linux";
#[cfg(target_os = "windows")]
pub const PLATFORM_NAME: &str = "windows";

#[cfg(not(target_os = "windows"))]
pub fn home_dir() -> std::path::PathBuf {
    if let Ok(h) = std::env::var("HOME") {
        return std::path::PathBuf::from(h);
    }
    std::path::PathBuf::from(".")
}

#[cfg(target_os = "windows")]
pub fn home_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("USERPROFILE") {
        return std::path::PathBuf::from(p);
    }
    if let Ok(d) = std::env::var("HOMEDRIVE") {
        if let Ok(p) = std::env::var("HOMEPATH") {
            return std::path::PathBuf::from(format!("{d}{p}"));
        }
    }
    std::path::PathBuf::from(".")
}
