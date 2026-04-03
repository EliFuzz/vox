use crate::settings::ModelId;
use serde_json::Value;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;
pub const REPO_OWNER: &str = "EliFuzz";
pub const REPO_NAME: &str = "vox";
const MODELS_TAG: &str = "models";
const MAX_RETRIES: u32 = 3;
const RETRY_BASE_DELAY: Duration = Duration::from_secs(2);
const API_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);
const PLATFORM: &str = crate::platform::PLATFORM_NAME;
#[derive(Debug)]
pub enum ModelError {
    Network(String),
    Io(std::io::Error),
    Extract(String),
    NotFound(String),
}
impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Network(s) => format!("network error: {s}"),
                Self::Io(e) => format!("io error: {e}"),
                Self::Extract(s) => format!("extraction error: {s}"),
                Self::NotFound(s) => format!("not found: {s}"),
            }
        )
    }
}
impl std::error::Error for ModelError {}
fn http_agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .build()
        .into()
}
fn ensure_asset(ready: bool, asset_name: &str, dest_subdir: &str) -> Result<(), ModelError> {
    if ready {
        return Ok(());
    }
    let models_dir = crate::settings::models_dir();
    fs::create_dir_all(&models_dir).map_err(ModelError::Io)?;
    let url = fetch_single_asset_url(asset_name)?;
    download_with_retry(&url, &models_dir.join(dest_subdir))
}
pub fn ensure_vad() -> Result<(), ModelError> {
    ensure_asset(
        crate::settings::vad_model_path().exists(),
        &format!("vad-{PLATFORM}.zip"),
        "vad",
    )
}
pub fn ensure_model(id: ModelId) -> Result<(), ModelError> {
    ensure_asset(
        crate::settings::model_ready(id),
        &format!("{}-{PLATFORM}.zip", id.as_str()),
        id.as_str(),
    )
}
fn fetch_single_asset_url(asset_name: &str) -> Result<String, ModelError> {
    let url =
        format!("https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/tags/{MODELS_TAG}");
    let agent = http_agent(API_TIMEOUT);
    let body: Value = agent
        .get(&url)
        .header("User-Agent", "vox-model-downloader/1.0")
        .call()
        .map_err(|e| ModelError::Network(e.to_string()))?
        .body_mut()
        .read_json()
        .map_err(|e| ModelError::Network(e.to_string()))?;
    let assets = body["assets"]
        .as_array()
        .ok_or_else(|| ModelError::NotFound("no assets in models release".into()))?;
    find_asset_download_url(assets, asset_name)
}
pub(crate) fn find_asset_download_url(assets: &[Value], name: &str) -> Result<String, ModelError> {
    assets
        .iter()
        .find(|a| a["name"].as_str() == Some(name))
        .and_then(|a| a["browser_download_url"].as_str())
        .map(String::from)
        .ok_or_else(|| ModelError::NotFound(format!("asset '{name}' not found in release")))
}
fn download_with_retry(url: &str, dest: &Path) -> Result<(), ModelError> {
    download_with_retry_using(fetch_and_extract, url, dest, RETRY_BASE_DELAY)
}
pub(crate) fn download_with_retry_using<F>(
    fetch: F,
    url: &str,
    dest: &Path,
    retry_base_delay: Duration,
) -> Result<(), ModelError>
where
    F: Fn(&str, &Path) -> Result<(), ModelError>,
{
    for attempt in 0..=MAX_RETRIES {
        match fetch(url, dest) {
            Ok(()) => return Ok(()),
            Err(e) if attempt < MAX_RETRIES => {
                cleanup_dir(dest);
                std::thread::sleep(retry_base_delay * 2u32.pow(attempt));
            }
            Err(e) => {
                cleanup_dir(dest);
                return Err(e);
            }
        }
    }
    unreachable!()
}
pub(crate) fn fetch_and_extract(url: &str, dest: &Path) -> Result<(), ModelError> {
    let agent = http_agent(DOWNLOAD_TIMEOUT);
    let mut data = Vec::new();
    agent
        .get(url)
        .header("User-Agent", "vox-model-downloader/1.0")
        .call()
        .map_err(|e| ModelError::Network(e.to_string()))?
        .body_mut()
        .as_reader()
        .read_to_end(&mut data)
        .map_err(ModelError::Io)?;
    validate_zip_magic(&data)?;
    extract_zip_to(&data, dest)
}
pub(crate) fn validate_zip_magic(data: &[u8]) -> Result<(), ModelError> {
    if data.len() < 4 || &data[..4] != b"PK\x03\x04" {
        return Err(ModelError::Extract(
            "downloaded file is not a valid zip archive".into(),
        ));
    }
    Ok(())
}
pub(crate) fn extract_zip_to(data: &[u8], dest: &Path) -> Result<(), ModelError> {
    fs::create_dir_all(dest).map_err(ModelError::Io)?;
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(data))
        .map_err(|e| ModelError::Extract(e.to_string()))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| ModelError::Extract(e.to_string()))?;
        let entry_path: PathBuf = match entry.enclosed_name() {
            Some(p) => PathBuf::from(p),
            None => continue,
        };
        let outpath = dest.join(entry_path);
        if entry.is_dir() {
            fs::create_dir_all(&outpath).map_err(ModelError::Io)?;
            continue;
        }
        if let Some(parent) = outpath.parent() {
            fs::create_dir_all(parent).map_err(ModelError::Io)?;
        }
        let mut out = fs::File::create(&outpath).map_err(ModelError::Io)?;
        std::io::copy(&mut entry, &mut out).map_err(ModelError::Io)?;
    }
    Ok(())
}
pub(crate) fn cleanup_dir(path: &Path) {
    if path.exists() {
        fs::remove_dir_all(path).ok();
    }
}
