use std::collections::HashSet;

use super::super::keyboard::all_shortcut_keys_held;

#[test]
fn single_key_held() {
    let shortcut = [rdev::Key::Function];
    let held = HashSet::from([rdev::Key::Function]);
    assert!(all_shortcut_keys_held(&held, &shortcut));
}

#[test]
fn single_key_not_held() {
    let shortcut = [rdev::Key::Function];
    let held = HashSet::new();
    assert!(!all_shortcut_keys_held(&held, &shortcut));
}

#[test]
fn multi_key_all_held() {
    let shortcut = [rdev::Key::Function, rdev::Key::ControlLeft];
    let held = HashSet::from([rdev::Key::Function, rdev::Key::ControlLeft]);
    assert!(all_shortcut_keys_held(&held, &shortcut));
}

#[test]
fn multi_key_partial_held() {
    let shortcut = [rdev::Key::Function, rdev::Key::ControlLeft];
    let held = HashSet::from([rdev::Key::Function]);
    assert!(!all_shortcut_keys_held(&held, &shortcut));
}

#[test]
fn multi_key_with_extra_keys_held() {
    let shortcut = [rdev::Key::Function, rdev::Key::ControlLeft];
    let held = HashSet::from([rdev::Key::Function, rdev::Key::ControlLeft, rdev::Key::KeyA]);
    assert!(all_shortcut_keys_held(&held, &shortcut));
}

#[test]
fn empty_shortcut_always_matches() {
    let held = HashSet::from([rdev::Key::KeyA]);
    assert!(all_shortcut_keys_held(&held, &[]));
}

#[test]
fn wrong_keys_held() {
    let shortcut = [rdev::Key::Function, rdev::Key::ControlLeft];
    let held = HashSet::from([rdev::Key::ShiftLeft, rdev::Key::KeyA]);
    assert!(!all_shortcut_keys_held(&held, &shortcut));
}

#[test]
fn three_key_combo() {
    let shortcut = [rdev::Key::ControlLeft, rdev::Key::ShiftLeft, rdev::Key::KeyA];
    let held = HashSet::from([rdev::Key::ControlLeft, rdev::Key::ShiftLeft, rdev::Key::KeyA]);
    assert!(all_shortcut_keys_held(&held, &shortcut));
}

#[test]
fn three_key_combo_missing_one() {
    let shortcut = [rdev::Key::ControlLeft, rdev::Key::ShiftLeft, rdev::Key::KeyA];
    let held = HashSet::from([rdev::Key::ControlLeft, rdev::Key::ShiftLeft]);
    assert!(!all_shortcut_keys_held(&held, &shortcut));
}
