use crate::settings::ModelId;
use std::path::Path;

#[test]
fn vad_model_path_resolves_to_mlmodelc() {
    let path = crate::settings::vad_model_path();
    assert!(path.ends_with(Path::new("vad/vad.mlmodelc")));
}

#[test]
fn asr_model_path_resolves_to_named_dir() {
    let path = crate::settings::model_path(ModelId::Multilingual);
    assert!(path.ends_with(Path::new("models/multilingual")));
}

#[test]
fn english_model_path_resolves_to_named_dir() {
    let path = crate::settings::model_path(ModelId::English);
    assert!(path.ends_with(Path::new("models/english")));
}

#[test]
fn models_dir_is_under_vox() {
    let vox = crate::settings::vox_dir();
    let models = crate::settings::models_dir();
    assert!(models.starts_with(&vox));
    assert!(models.ends_with("models"));
}
