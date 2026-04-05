use crate::settings::{resolve_shortcut, Settings, DEFAULT_SHORTCUT};

#[test]
fn parses_multi_key_array() {
    let s: Settings =
        serde_json::from_str(r#"{"shortcut":["Function","ControlLeft"]}"#).unwrap();
    assert_eq!(
        s.shortcut,
        Some(vec![rdev::Key::Function, rdev::Key::ControlLeft])
    );
}

#[test]
fn null_shortcut_returns_none() {
    let s: Settings = serde_json::from_str(r#"{"shortcut":null}"#).unwrap();
    assert_eq!(s.shortcut, None);
}

#[test]
fn missing_shortcut_returns_none() {
    let s: Settings = serde_json::from_str(r#"{}"#).unwrap();
    assert_eq!(s.shortcut, None);
}

#[test]
fn resolve_shortcut_uses_default_when_none() {
    assert_eq!(resolve_shortcut(&None), DEFAULT_SHORTCUT.to_vec());
}

#[test]
fn resolve_shortcut_uses_default_when_empty() {
    assert_eq!(resolve_shortcut(&Some(vec![])), DEFAULT_SHORTCUT.to_vec());
}

#[test]
fn resolve_shortcut_uses_provided_keys() {
    let keys = vec![rdev::Key::ShiftLeft, rdev::Key::KeyA];
    assert_eq!(resolve_shortcut(&Some(keys.clone())), keys);
}
