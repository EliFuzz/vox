use crate::model_download::{
    download_with_retry_using, find_asset_download_url, validate_zip_magic, ModelError,
};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

fn test_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("test_artifacts/model_download/{name}"));
    if dir.exists() {
        fs::remove_dir_all(&dir).ok();
    }
    dir
}

#[test]
fn find_asset_returns_correct_download_url() {
    let assets = serde_json::json!([
        {
            "name": "multilingual-macos.zip",
            "browser_download_url": "https://github.com/EliFuzz/vox/releases/download/models/multilingual-macos.zip"
        },
        {
            "name": "vad-macos.zip",
            "browser_download_url": "https://github.com/EliFuzz/vox/releases/download/models/vad-macos.zip"
        }
    ]);
    let assets = assets.as_array().unwrap();
    let url = find_asset_download_url(assets, "multilingual-macos.zip").unwrap();
    assert_eq!(
        url,
        "https://github.com/EliFuzz/vox/releases/download/models/multilingual-macos.zip"
    );
}

#[test]
fn find_asset_returns_not_found_for_missing() {
    let assets: Vec<serde_json::Value> = vec![];
    let result = find_asset_download_url(&assets, "missing.zip");
    assert!(matches!(result, Err(ModelError::NotFound(_))));
}

#[test]
fn find_asset_returns_not_found_when_url_absent() {
    let assets = serde_json::json!([{"name": "file.zip"}]);
    let assets = assets.as_array().unwrap();
    let result = find_asset_download_url(assets, "file.zip");
    assert!(matches!(result, Err(ModelError::NotFound(_))));
}

#[test]
fn network_failure_propagated_as_network_error() {
    let dir = test_dir("network_error");
    let result = download_with_retry_using(
        |_url, _dest| Err(ModelError::Network("connection refused".to_string())),
        "https://192.0.2.1/unreachable.zip",
        &dir,
        Duration::ZERO,
    );
    assert!(matches!(result, Err(ModelError::Network(_))));
    assert!(!dir.exists());
}

#[test]
fn corrupted_download_propagated_as_extract_error() {
    let dir = test_dir("corrupted_download");
    let result = download_with_retry_using(
        |_url, dest| {
            fs::create_dir_all(dest).map_err(ModelError::Io)?;
            validate_zip_magic(b"corrupted data")
        },
        "https://example.com/test.zip",
        &dir,
        Duration::ZERO,
    );
    assert!(matches!(result, Err(ModelError::Extract(_))));
    assert!(!dir.exists());
}

#[test]
fn io_failure_propagated_as_io_error() {
    let dir = test_dir("io_error");
    let result = download_with_retry_using(
        |_url, _dest| {
            Err(ModelError::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "simulated permission denied",
            )))
        },
        "https://example.com/test.zip",
        &dir,
        Duration::ZERO,
    );
    assert!(matches!(result, Err(ModelError::Io(_))));
    assert!(!dir.exists());
}

#[test]
#[ignore = "requires network access and downloads large model files"]
fn full_model_download_and_extraction_succeeds() {
    let models_dir = PathBuf::from("test_artifacts/full_download");
    fs::create_dir_all(&models_dir).unwrap();
    let ml_dest = models_dir.join("multilingual");
    let vad_dest = models_dir.join("vad");
    std::thread::scope(|s| {
        s.spawn(|| {
            crate::model_download::ensure_model(crate::settings::ModelId::Multilingual)
                .expect("multilingual download failed")
        });
        s.spawn(|| crate::model_download::ensure_vad().expect("vad download failed"));
    });
    assert!(ml_dest.join("encoder.mlmodelc").exists());
    assert!(vad_dest.join("vad.mlmodelc").exists());
    fs::remove_dir_all(&models_dir).ok();
}
