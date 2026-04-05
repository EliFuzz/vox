use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelId {
    #[default]
    Multilingual,
    English,
    EnglishSmall,
}
impl ModelId {
    pub const ALL: &[ModelId] = &[
        ModelId::Multilingual,
        ModelId::English,
        ModelId::EnglishSmall,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Multilingual => "multilingual",
            Self::English => "english",
            Self::EnglishSmall => "english_small",
        }
    }
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Multilingual => "Multilingual",
            Self::English => "English",
            Self::EnglishSmall => "English Small",
        }
    }
}
#[derive(Deserialize, Serialize, Default, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    PushToTalk,
    ToggleToTalk,
}
pub const DEFAULT_SHORTCUT: &[rdev::Key] = &[rdev::Key::Function, rdev::Key::ControlLeft];

#[derive(Deserialize, Serialize, Clone, Default)]
pub struct Settings {
    #[serde(default)]
    pub model: ModelId,
    #[serde(default)]
    pub mode: Mode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shortcut: Option<Vec<rdev::Key>>,
    pub input_device: Option<String>,
}

pub fn resolve_shortcut(shortcut: &Option<Vec<rdev::Key>>) -> Vec<rdev::Key> {
    shortcut
        .as_ref()
        .filter(|keys| !keys.is_empty())
        .cloned()
        .unwrap_or_else(|| DEFAULT_SHORTCUT.to_vec())
}
pub fn vox_dir() -> PathBuf {
    crate::platform::home_dir().join(".vox")
}
pub fn models_dir() -> PathBuf {
    vox_dir().join("models")
}
pub fn model_path(id: ModelId) -> PathBuf {
    models_dir().join(id.as_str())
}
pub fn vad_model_path() -> PathBuf {
    models_dir().join("vad").join("vad.mlmodelc")
}
pub fn load() -> Settings {
    let path = vox_dir().join("settings.json");
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}
pub fn save(settings: &Settings) {
    let path = vox_dir().join("settings.json");
    std::fs::create_dir_all(vox_dir()).ok();
    if let Ok(json) = serde_json::to_string_pretty(settings) {
        std::fs::write(&path, json).ok();
    }
}
pub fn model_ready(id: ModelId) -> bool {
    let dir = model_path(id);
    crate::asr::ready_check_files(id)
        .iter()
        .all(|f| dir.join(f).exists())
}
