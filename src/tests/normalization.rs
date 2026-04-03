use crate::settings::ModelId;
use std::collections::HashMap;

fn load_wav_samples(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect()
}

fn run_e2e(model_id: ModelId, audio_file: &str, expected: &str, label: &str) {
    let model_dir = crate::settings::model_path(model_id);
    if !crate::settings::model_ready(model_id) {
        return;
    }
    let samples = load_wav_samples(std::path::Path::new(audio_file));
    let mut asr = crate::asr::Asr::new(model_id, &model_dir);
    let raw = asr.transcribe(&samples);
    let normalized = crate::normalization::normalize_with_punctuation(&raw);
    assert!(
        normalized.eq_ignore_ascii_case(expected),
        "{label} output: raw={raw:?}, normalized={normalized:?}"
    );
}

#[test]
#[cfg(target_os = "macos")]
fn parakeet_date_pipeline_runs_end_to_end() {
    run_e2e(
        ModelId::Multilingual,
        "asr/audio/date.wav",
        "January 5, 2025.",
        "parakeet",
    );
}

#[test]
#[cfg(target_os = "macos")]
fn parakeet_yep_pipeline_runs_end_to_end() {
    run_e2e(
        ModelId::Multilingual,
        "asr/audio/yep.wav",
        "Yep.",
        "parakeet",
    );
}

#[test]
#[cfg(target_os = "macos")]
fn moonshine_date_pipeline_runs_end_to_end() {
    run_e2e(
        ModelId::EnglishSmall,
        "asr/audio/date.wav",
        "January 5, 2025.",
        "moonshine",
    );
}

#[test]
#[cfg(target_os = "macos")]
fn moonshine_yep_pipeline_runs_end_to_end() {
    run_e2e(
        ModelId::EnglishSmall,
        "asr/audio/yep.wav",
        "Yeah.",
        "moonshine",
    );
}

#[test]
#[cfg(target_os = "macos")]
fn english_date_pipeline_runs_end_to_end() {
    run_e2e(
        ModelId::English,
        "asr/audio/date.wav",
        "January 5th, 2025.",
        "english",
    );
}

#[test]
#[cfg(target_os = "macos")]
fn english_yep_pipeline_runs_end_to_end() {
    run_e2e(ModelId::English, "asr/audio/yep.wav", "Yep.", "english");
}

#[test]
fn debug_normalization_outputs() {
    let inputs = HashMap::from([
        ("January fifth, twenty twenty five.", "January 5, 2025."),
        ("January fifth, twenty twenty five", "January 5, 2025"),
        ("January fifth twenty twenty five.", "January 5 2025."),
        ("January fifth twenty twenty five", "January 5 2025"),
        ("twenty twenty five.", "2025."),
        ("twenty twenty five", "2025"),
    ]);

    for (input, expected) in inputs {
        let actual = crate::normalization::normalize_with_punctuation(input);
        assert_eq!(
            actual, expected,
            "input={input:?}, expected={expected:?}, actual={actual:?}"
        );
    }
}
