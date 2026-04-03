use crate::model_download::{
    cleanup_dir, download_with_retry_using, extract_zip_to, validate_zip_magic, ModelError,
};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn test_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("test_artifacts/model_download/{name}"));
    if dir.exists() {
        fs::remove_dir_all(&dir).ok();
    }
    dir
}

fn make_test_zip(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut buf = Vec::new();
    let cursor = std::io::Cursor::new(&mut buf);
    let mut writer = zip::ZipWriter::new(cursor);
    let options =
        zip::write::FileOptions::<()>::default().compression_method(zip::CompressionMethod::Stored);
    for (name, data) in files {
        writer.start_file(*name, options).unwrap();
        writer.write_all(data).unwrap();
    }
    writer.finish().unwrap();
    buf
}

#[test]
fn zip_extraction_creates_expected_files() {
    let dir = test_dir("zip_extraction");
    let zip_data = make_test_zip(&[("encoder.mlmodelc", b"model"), ("vocab.json", b"{}")]);
    extract_zip_to(&zip_data, &dir).unwrap();
    assert!(dir.join("encoder.mlmodelc").exists());
    assert!(dir.join("vocab.json").exists());
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn zip_extraction_creates_nested_directories() {
    let dir = test_dir("zip_nested");
    let zip_data = make_test_zip(&[("subdir/file.bin", b"data")]);
    extract_zip_to(&zip_data, &dir).unwrap();
    assert!(dir.join("subdir/file.bin").exists());
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn corrupted_zip_returns_extract_error() {
    let dir = test_dir("corrupted");
    let result = extract_zip_to(b"this is not a zip", &dir);
    assert!(matches!(result, Err(ModelError::Extract(_))));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn zip_magic_rejects_invalid_data() {
    assert!(validate_zip_magic(b"not a zip").is_err());
    assert!(validate_zip_magic(b"").is_err());
    assert!(validate_zip_magic(b"PK").is_err());
}

#[test]
fn zip_magic_accepts_valid_header() {
    assert!(validate_zip_magic(b"PK\x03\x04extra").is_ok());
}

#[test]
fn retry_succeeds_after_transient_failures() {
    let dir = test_dir("retry_success");
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = call_count.clone();
    let result = download_with_retry_using(
        move |_url, dest| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                return Err(ModelError::Network("transient".to_string()));
            }
            fs::create_dir_all(dest).map_err(ModelError::Io)?;
            fs::write(dest.join("encoder.mlmodelc"), b"ok").map_err(ModelError::Io)
        },
        "https://example.com/test.zip",
        &dir,
        Duration::ZERO,
    );
    assert!(result.is_ok(), "expected success after transient failures");
    assert_eq!(call_count.load(Ordering::SeqCst), 3);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn retry_exhausted_returns_last_error() {
    let dir = test_dir("retry_exhausted");
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = call_count.clone();
    let result = download_with_retry_using(
        move |_url, _dest| {
            cc.fetch_add(1, Ordering::SeqCst);
            Err(ModelError::Network("always fails".to_string()))
        },
        "https://example.com/test.zip",
        &dir,
        Duration::ZERO,
    );
    assert!(matches!(result, Err(ModelError::Network(_))));
    assert_eq!(call_count.load(Ordering::SeqCst), 4);
    assert!(
        !dir.exists(),
        "partial directory must be cleaned up after exhausted retries"
    );
}

#[test]
fn partial_directory_cleaned_up_on_failure() {
    let dir = test_dir("partial_cleanup");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("partial.bin"), b"incomplete").unwrap();
    let result = download_with_retry_using(
        |_url, _dest| Err(ModelError::Network("fail".to_string())),
        "https://example.com/test.zip",
        &dir,
        Duration::ZERO,
    );
    assert!(result.is_err());
    assert!(
        !dir.exists(),
        "partial directory must be removed after failure"
    );
}

#[test]
fn cleanup_removes_directory() {
    let dir = test_dir("cleanup_removes");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("file.bin"), b"data").unwrap();
    cleanup_dir(&dir);
    assert!(!dir.exists());
}

#[test]
fn cleanup_noop_when_absent() {
    let dir = test_dir("cleanup_absent");
    assert!(!dir.exists());
    cleanup_dir(&dir);
}
